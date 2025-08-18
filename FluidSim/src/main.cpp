#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

extern "C" void runFillGradientKernel(uchar4* devPtr, int width, int height, float time);

#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

const int TEX_WIDTH = 800;
const int TEX_HEIGHT = 600;

GLuint pbo = 0;
struct cudaGraphicsResource* cuda_pbo_resource;
GLuint texID = 0;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void createPBO() {
    if (pbo) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
    }
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, TEX_WIDTH * TEX_HEIGHT * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
}

void createTexture() {
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, TEX_WIDTH, TEX_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
}

void runCuda(float time) {
    uchar4* devPtr;
    size_t size;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cuda_pbo_resource));

    runFillGradientKernel(devPtr, TEX_WIDTH, TEX_HEIGHT, time);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

void renderTexture() {
    glClear(GL_COLOR_BUFFER_BIT);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, texID);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEX_WIDTH, TEX_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Render a fullscreen quad with fixed-function pipeline (for simplicity)
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1, -1);
        glTexCoord2f(1, 0); glVertex2f( 1, -1);
        glTexCoord2f(1, 1); glVertex2f( 1,  1);
        glTexCoord2f(0, 1); glVertex2f(-1,  1);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

int main_2() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);  // Use OpenGL 3.x for compatibility with fixed pipeline fallback
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);  // Use compatibility profile

    GLFWwindow* window = glfwCreateWindow(TEX_WIDTH, TEX_HEIGHT, "CUDA + OpenGL Interop Test", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    }

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glViewport(0, 0, TEX_WIDTH, TEX_HEIGHT);

    createPBO();
    createTexture();

    while (!glfwWindowShouldClose(window)) {
        float time = (float)glfwGetTime();

        runCuda(time);
        renderTexture();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texID);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
