project(Roboman-SDM)
cmake_minimum_required(VERSION 2.8.12)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

# original flags moved to WIN32 section
if (WIN32)
        #MinGW编译的时候，添加-static选项，去除对其他三个标准库的依赖，单独添加-static-libstdc++，只去除对libstdc++标准的依赖
        SET(CMAKE_C_FLAGS   "-fPIC -static")
        SET(CMAKE_CXX_FLAGS "-fPIC -static")

        # g++ needs a compiler flag to enable C++11 support
        include(CheckCXXCompilerFlag)
        CHECK_CXX_COMPILER_FLAG(-std=c++11 HAS_CXX11_FLAG)
        if (HAS_CXX11_FLAG)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
                message("..........................................")
        else()
                message("WARNING: THERE IS NO CXX11 FLAG...........")
        endif()
    
        SET(OpenCVSDK_Path ${CMAKE_CURRENT_LIST_DIR}/OpenCVSDK)

        SET(OpenCV_LIBS ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_contrib248.a        ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_legacy248.a
                        ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_stitching248.a      ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_superres248.a
                        ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_ocl248.a            ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_objdetect248.a
                        ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_ml248.a             ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_ts248.a
                        ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_videostab248.a      ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_video248.a
                        ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_photo248.a          ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_calib3d248.a
                        ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_features2d248.a     ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_highgui248.a
                        ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_imgproc248.a        ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_flann248.a
                        ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_ts248.a             ${OpenCVSDK_Path}/x86/mingw/staticlib/libopencv_core248.a

                        ${OpenCVSDK_Path}/x86/mingw/staticlib/libzlib.a
                        ${OpenCVSDK_Path}/x86/mingw/staticlib/liblibjpeg.a
                        ${OpenCVSDK_Path}/x86/mingw/staticlib/liblibpng.a
                        ${OpenCVSDK_Path}/x86/mingw/staticlib/liblibtiff.a
                        ${OpenCVSDK_Path}/x86/mingw/staticlib/liblibjasper.a
                        ${OpenCVSDK_Path}/x86/mingw/staticlib/libIlmImf.a

                        -lcomctl32 -lvfw32
                        )
        include_directories(${OpenCVSDK_Path}/include)

endif (WIN32)

# OSX only config (might work UNIX too, not tested)
if(APPLE)
        # add c++11 flags
        add_compile_options(-std=c++11)
        # use OpenCV dynamic libraries instead of static: there's an issue compiling OpenCV static libraries (ld: library not found for -lcrt0.o)
        find_package( OpenCV REQUIRED )
endif(APPLE)

include_directories("src/include")
aux_source_directory("src/include" SOURCES_FILE)

#ADD_LIBRARY(SRC ${SOURCES_FILE})

#ADD_EXECUTABLE(Train-SDM src/train_model.cpp)
#TARGET_LINK_LIBRARIES(Train-SDM ${OpenCV_LIBS})

ADD_EXECUTABLE(Roboman-SDM-TEST src/test_model.cpp)
TARGET_LINK_LIBRARIES(Roboman-SDM-TEST ${OpenCV_LIBS})




