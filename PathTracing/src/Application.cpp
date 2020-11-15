#include <cstdlib> 
#include <cstdio> 
#include <cmath> 
#include <fstream> 
#include <vector>
#include <iostream> 
#include <cassert>
#include <random>
#include <future>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

# define M_PI 3.14159265358979323846
    
#include "geometry.h"
#include "ThreadPool.h"
#include "Window.h"

enum RayType { kPrimaryRay, kShadowRay };
static const float kInfinity = std::numeric_limits<float>::max();
static const Vec3f kDefaultBackgroundColor = (255);

//structures

//classes
bool solveQuadratic(const float& a, const float& b, const float& c, float& x0, float& x1)
{
    float discr = b * b - 4 * a * c;
    if (discr < 0) return false;
    else if (discr == 0) {
        x0 = x1 = -0.5 * b / a;
    }
    else {
        float q = (b > 0) ?
            -0.5 * (b + sqrt(discr)) :
            -0.5 * (b - sqrt(discr));
        x0 = q / a;
        x1 = c / q;
    }

    return true;
}

class Sphere
{
public:
    Vec3f center;
    float radius, radius2;
    Vec3f surfaceColor;

    Sphere
    (
        const Vec3f& c,
        const float& r,
        const Vec3f& sc
    ) : center(c), radius(r), radius2(r* r), surfaceColor(sc) {}

    bool intersect(const Vec3f& rayorig, const Vec3f& raydir, float& tnear) const
    {
        float t0, t1; // solutions for t if the ray intersects 
        // analytic solution
        Vec3f L = rayorig - center;
        float a = raydir.dotProduct(raydir);
        float b = 2 * raydir.dotProduct(L);
        float c = L.dotProduct(L) - radius2;

        if (!solveQuadratic(a, b, c, t0, t1)) return false;

        if (t0 > t1) std::swap(t0, t1);

        if (t0 < 0) {
            t0 = t1; // if t0 is negative, let's use t1 instead 
            if (t0 < 0) return false; // both t0 and t1 are negative 
        }

        tnear = t0;

        return true;
    }

    void getSurfaceProperties(
        const Vec3f& hitp,
        const Vec3f& dir,
        Vec3f& hitn) const
    {
        hitn = hitp - center;
        hitn.normalize();
    }
};

struct Options
{
    uint32_t width = 640;
    uint32_t height = 480;
    float fov = 90;
    Vec3f backgroundColor = kDefaultBackgroundColor;
    Matrix44f cameraToWorld;
    float bias = 0.0001;
    uint32_t maxDepth = 2;
    uint32_t samples = 1;;
};

struct IsectInfo
{
    const Sphere* hitObject = nullptr;
    float tNear = kInfinity;
    uint32_t index = 0;
};

class Light
{
public:
    Light(const Matrix44f& l2w, const Vec3f& c = 1, const float& i = 1) : lightToWorld(l2w), color(c), intensity(i) {}
    virtual ~Light() {}
    virtual void illuminate(const Vec3f& P, Vec3f&, Vec3f&, float&) const = 0;
    Vec3f color;
    float intensity;
    Matrix44f lightToWorld;
};

class PointLight : public Light
{
    Vec3f pos;
public:
    PointLight(const Matrix44f& l2w, const Vec3f& c = 1, const float& i = 1) : Light(l2w, c, i)
    {
        l2w.multVecMatrix(Vec3f(0), pos);
    }
    // P: is the shaded point
    void illuminate(const Vec3f& P, Vec3f& lightDir, Vec3f& lightIntensity, float& distance) const
    {
        lightDir = (P - pos);
        float r2 = lightDir.norm();
        distance = sqrt(r2);
        lightDir.x /= distance, lightDir.y /= distance, lightDir.z /= distance;
        // avoid division by 0
        lightIntensity = color * intensity / (4 * M_PI * r2);
    }
};

class DistantLight : public Light
{
    Vec3f dir;
public:
    DistantLight(const Matrix44f& l2w, const Vec3f& c = 1, const float& i = 1) : Light(l2w, c, i)
    {
        l2w.multDirMatrix(Vec3f(0, 0, -1), dir);
        dir.normalize(); // in case the matrix scales the light 
    }
    void illuminate(const Vec3f& P, Vec3f& lightDir, Vec3f& lightIntensity, float& distance) const
    {
        lightDir = dir;
        lightIntensity = color * intensity;
        distance = kInfinity;
    }
};

//functions

bool trace(
    const Vec3f& orig, const Vec3f& dir,
    const std::vector<std::unique_ptr<Sphere>>& spheres,
    IsectInfo& isect,
    RayType raytype = kPrimaryRay) //args
{
    isect.hitObject = nullptr;

    for (uint32_t i = 0; i < spheres.size(); i++)
    {
        float tnear = kInfinity;
        uint32_t index = 0;

        if (spheres[i]->intersect(orig, dir, tnear) && tnear < isect.tNear)
        {
            isect.hitObject = spheres[i].get();
            isect.tNear = tnear;
            isect.index = index;
        }
    }
    return (isect.hitObject != nullptr);
}

void createCoordinateSystem(const Vec3f& N, Vec3f& Nt, Vec3f& Nb)
{
    if (std::fabs(N.x) > std::fabs(N.y))
        Nt = Vec3f(N.z, 0, -N.x) / sqrtf(N.x * N.x + N.z * N.z);
    else
        Nt = Vec3f(0, -N.z, N.y) / sqrtf(N.y * N.y + N.z * N.z);
    Nb = N.crossProduct(Nt);
}

std::default_random_engine generator;
std::uniform_real_distribution<float> distribution(0, 1);

Vec3f uniformSampleHemisphere(const float& r1, const float& r2)
{
    // cos(theta) = u1 = y
    // cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = srtf(1 - cos^2(theta))
    float sinTheta = sqrtf(1 - r1 * r1);
    float phi = 2 * M_PI * r2;
    float x = sinTheta * cosf(phi);
    float z = sinTheta * sinf(phi);
    return Vec3f(x, r1, z);
}

Vec3f cast(
    const Vec3f& rayorig,
    const Vec3f& raydir,
    const std::vector<std::unique_ptr<Sphere>>& spheres,
    const std::vector<std::unique_ptr<Light>>& lights,
    const Options options,
    const uint32_t& depth = 0)
{
    if (depth > options.maxDepth) return 0;
    IsectInfo isect;
    Vec3f hit_color = 0;

    if (trace(rayorig, raydir, spheres, isect))
    {
        Vec3f hitp = rayorig + raydir * isect.tNear;
        Vec3f hitn;
        isect.hitObject->getSurfaceProperties(hitp, raydir, hitn);

        //diffuse direct lightning
        Vec3f direct_l = 0;
        for (uint32_t i = 0; i < lights.size(); i++)
        {
            Vec3f l_dir, l_intensity;
            IsectInfo isectShad;
            lights[i]->illuminate(hitp, l_dir, l_intensity, isectShad.tNear);
            bool vis = !trace(hitp + hitn * options.bias, -l_dir, spheres, isectShad, kShadowRay);
            direct_l = vis * l_intensity * std::max(0.f, hitn.dotProduct(-l_dir));
        }

        //indirect lighting
        Vec3f indirect_l = 0;

        Vec3f Nt, Nb;
        createCoordinateSystem(hitn, Nt, Nb);
        float pdf = 1 / (2 * M_PI);

        for (uint32_t n = 0; n < options.samples; ++n) {
            float r1 = distribution(generator);
            float r2 = distribution(generator);

            Vec3f sample = uniformSampleHemisphere(r1, r2);
            Vec3f sampleWorld(
                sample.x * Nb.x + sample.y * hitn.x + sample.z * Nt.x,
                sample.x * Nb.y + sample.y * hitn.y + sample.z * Nt.y,
                sample.x * Nb.z + sample.y * hitn.z + sample.z * Nt.z);

            indirect_l += r1 * cast(hitp + sampleWorld * options.bias, sampleWorld, spheres, lights, options, depth + 1) / pdf;
        }
        // divide by N
        indirect_l /= (float)options.samples;
        //indirect_l = 0;

        hit_color = (direct_l / M_PI + 2 * indirect_l) * 0.18 * isect.hitObject->surfaceColor;
    }
    else
        hit_color = 0.4;

    return hit_color;

}

static std::vector<std::promise<Vec3f>> s_promises;
static std::vector<std::future<Vec3f>> s_futures;

void render(
    const Options& options,
    const std::vector<std::unique_ptr<Sphere>>& spheres,
    const std::vector<std::unique_ptr<Light>>& lights
)
{
    unsigned width = options.width, height = options.height;
    Vec3f* image = new Vec3f[width * height], *pixel = image;
    float invWidth = 1 / float(width), invHeight = 1 / float(height);
    float fov = options.fov, aspectratio = width * invHeight;
    float angle = tan(M_PI * 0.5 * fov / 180.);
    Vec3f v_init(0);
    

    //cast rays
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x, ++pixel)
        {
            float xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectratio;
            float yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle;
        
            Vec3f raydir(xx, yy, -1);
            raydir.normalize();
            
            *pixel = cast(Vec3f(0), raydir, spheres, lights, options, 0);
            //pool.enqueue([pixel, &raydir, &spheres, &lights, options]() {*pixel = cast(Vec3f(0), raydir, spheres, lights, options, 0); });

            cv::circle(Window::image, cv::Point(x, y), 0, cv::Scalar(pixel->x * 255, pixel->y * 255, pixel->z * 255));
        }

    // Save result to a PPM image (keep these flags if you compile under Windows)
    /*
    std::ofstream ofs("final_renders/16_1_IS.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (unsigned i = 0; i < width * height; ++i) {
        ofs << (unsigned char)(std::min(float(1), image[i].x) * 255) <<
            (unsigned char)(std::min(float(1), image[i].y) * 255) <<
            (unsigned char)(std::min(float(1), image[i].z) * 255);
    }
    ofs.close();*/

    delete[] image;
}

int main(int argc, char** argv)
{
    std::vector<std::unique_ptr<Sphere>> spheres;
    std::vector<std::unique_ptr<Light>> lights;
    Options options;

    // aliasing example
    options.fov = 39.89;
    options.width = 512;
    options.height = 512;
    options.backgroundColor = Vec3f(0.8, 0.8, 1);
    options.cameraToWorld = Matrix44f(0.965926, 0, -0.258819, 0, 0.0066019, 0.999675, 0.0246386, 0, 0.258735, -0.0255078, 0.965612, 0, 0.764985, 0.791882, 5.868275, 1);
    options.samples = 16;
    options.maxDepth = 1;

    Matrix44f l2w(0.916445, -0.218118, 0.335488, 0, 0.204618, -0.465058, -0.861309, 0, 0.343889, 0.857989, -0.381569, 0, 0, 0, 0, 1);
    
    //world config
    spheres.push_back(std::unique_ptr<Sphere>(new Sphere(Vec3f(0.0, -23, -20), 20, Vec3f(1.00, 0.32, 0.36))));
    spheres.push_back(std::unique_ptr<Sphere>(new Sphere(Vec3f(0.0, 0, -20), 3, Vec3f(0.70, 0.70, 0.70))));
    spheres.push_back(std::unique_ptr<Sphere>(new Sphere(Vec3f(5.0, -1, -15), 2, Vec3f(0.90, 0.76, 0.46))));
    spheres.push_back(std::unique_ptr<Sphere>(new Sphere(Vec3f(5.0, 0, -25), 3, Vec3f(0.65, 0.77, 0.97))));
    spheres.push_back(std::unique_ptr<Sphere>(new Sphere(Vec3f(-5.5, 0, -15), 3, Vec3f(0.90, 0.20, 0.90))));
    lights.push_back(std::unique_ptr<Light>(new DistantLight(l2w, 1, 4)));

    //window config
    Window::init("Ray Tracer result", options.width, options.height, 60);

    std::clock_t start = std::clock();
    render(options, spheres, lights);
    std::clock_t end = std::clock();

    Window::update();

    printf("Excecution time: %.2fs", (end - start) / (double)CLOCKS_PER_SEC);

    // create thread pool with 4 worker threads
    ThreadPool pool(4);

    // enqueue and store future
    auto result = pool.enqueue([](int answer) { return answer; }, 42);

    // get result from future
    std::cout << result.get() << std::endl;

    std::cin.get();
    return 0;
}