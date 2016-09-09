#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "zlib" for configuration "Release"
set_property(TARGET zlib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(zlib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libzlib.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS zlib )
list(APPEND _IMPORT_CHECK_FILES_FOR_zlib "${_IMPORT_PREFIX}/x64/mingw/staticlib/libzlib.a" )

# Import target "libtiff" for configuration "Release"
set_property(TARGET libtiff APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libtiff PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "zlib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/liblibtiff.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS libtiff )
list(APPEND _IMPORT_CHECK_FILES_FOR_libtiff "${_IMPORT_PREFIX}/x64/mingw/staticlib/liblibtiff.a" )

# Import target "libjpeg" for configuration "Release"
set_property(TARGET libjpeg APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libjpeg PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/liblibjpeg.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS libjpeg )
list(APPEND _IMPORT_CHECK_FILES_FOR_libjpeg "${_IMPORT_PREFIX}/x64/mingw/staticlib/liblibjpeg.a" )

# Import target "libjasper" for configuration "Release"
set_property(TARGET libjasper APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libjasper PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/liblibjasper.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS libjasper )
list(APPEND _IMPORT_CHECK_FILES_FOR_libjasper "${_IMPORT_PREFIX}/x64/mingw/staticlib/liblibjasper.a" )

# Import target "libpng" for configuration "Release"
set_property(TARGET libpng APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libpng PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "zlib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/liblibpng.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS libpng )
list(APPEND _IMPORT_CHECK_FILES_FOR_libpng "${_IMPORT_PREFIX}/x64/mingw/staticlib/liblibpng.a" )

# Import target "IlmImf" for configuration "Release"
set_property(TARGET IlmImf APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(IlmImf PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "zlib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libIlmImf.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS IlmImf )
list(APPEND _IMPORT_CHECK_FILES_FOR_IlmImf "${_IMPORT_PREFIX}/x64/mingw/staticlib/libIlmImf.a" )

# Import target "opencv_core" for configuration "Release"
set_property(TARGET opencv_core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "zlib;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_core248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_core )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_core "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_core248.a" )

# Import target "opencv_flann" for configuration "Release"
set_property(TARGET opencv_flann APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_flann PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_flann248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_flann )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_flann "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_flann248.a" )

# Import target "opencv_imgproc" for configuration "Release"
set_property(TARGET opencv_imgproc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_imgproc PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_imgproc248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_imgproc )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_imgproc "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_imgproc248.a" )

# Import target "opencv_highgui" for configuration "Release"
set_property(TARGET opencv_highgui APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_highgui PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;opencv_imgproc;stdc++;zlib;libjpeg;libpng;libtiff;libjasper;IlmImf;comctl32;gdi32;ole32;setupapi;ws2_32;avifil32;avicap32;winmm;msvfw32"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_highgui248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_highgui )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_highgui "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_highgui248.a" )

# Import target "opencv_features2d" for configuration "Release"
set_property(TARGET opencv_features2d APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_features2d PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;opencv_flann;opencv_imgproc;opencv_highgui;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_features2d248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_features2d )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_features2d "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_features2d248.a" )

# Import target "opencv_calib3d" for configuration "Release"
set_property(TARGET opencv_calib3d APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_calib3d PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;opencv_flann;opencv_imgproc;opencv_highgui;opencv_features2d;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_calib3d248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_calib3d )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_calib3d "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_calib3d248.a" )

# Import target "opencv_ml" for configuration "Release"
set_property(TARGET opencv_ml APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_ml PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_ml248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_ml )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_ml "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_ml248.a" )

# Import target "opencv_video" for configuration "Release"
set_property(TARGET opencv_video APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_video PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;opencv_imgproc;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_video248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_video )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_video "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_video248.a" )

# Import target "opencv_legacy" for configuration "Release"
set_property(TARGET opencv_legacy APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_legacy PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;opencv_flann;opencv_imgproc;opencv_highgui;opencv_features2d;opencv_calib3d;opencv_ml;opencv_video;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_legacy248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_legacy )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_legacy "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_legacy248.a" )

# Import target "opencv_objdetect" for configuration "Release"
set_property(TARGET opencv_objdetect APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_objdetect PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;opencv_imgproc;opencv_highgui;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_objdetect248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_objdetect )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_objdetect "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_objdetect248.a" )

# Import target "opencv_photo" for configuration "Release"
set_property(TARGET opencv_photo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_photo PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;opencv_imgproc;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_photo248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_photo )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_photo "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_photo248.a" )

# Import target "opencv_gpu" for configuration "Release"
set_property(TARGET opencv_gpu APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_gpu PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;opencv_flann;opencv_imgproc;opencv_highgui;opencv_features2d;opencv_calib3d;opencv_ml;opencv_video;opencv_legacy;opencv_objdetect;opencv_photo;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_gpu248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_gpu )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_gpu "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_gpu248.a" )

# Import target "opencv_ocl" for configuration "Release"
set_property(TARGET opencv_ocl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_ocl PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;opencv_flann;opencv_imgproc;opencv_highgui;opencv_features2d;opencv_calib3d;opencv_ml;opencv_objdetect;opencv_video;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_ocl248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_ocl )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_ocl "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_ocl248.a" )

# Import target "opencv_nonfree" for configuration "Release"
set_property(TARGET opencv_nonfree APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_nonfree PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;opencv_flann;opencv_imgproc;opencv_highgui;opencv_features2d;opencv_calib3d;opencv_ml;opencv_video;opencv_legacy;opencv_objdetect;opencv_photo;opencv_gpu;opencv_ocl;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_nonfree248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_nonfree )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_nonfree "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_nonfree248.a" )

# Import target "opencv_contrib" for configuration "Release"
set_property(TARGET opencv_contrib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_contrib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;opencv_flann;opencv_imgproc;opencv_highgui;opencv_features2d;opencv_calib3d;opencv_ml;opencv_video;opencv_legacy;opencv_objdetect;opencv_photo;opencv_gpu;opencv_ocl;opencv_nonfree;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_contrib248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_contrib )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_contrib "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_contrib248.a" )

# Import target "opencv_stitching" for configuration "Release"
set_property(TARGET opencv_stitching APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_stitching PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;opencv_flann;opencv_imgproc;opencv_highgui;opencv_features2d;opencv_calib3d;opencv_ml;opencv_video;opencv_legacy;opencv_objdetect;opencv_photo;opencv_gpu;opencv_ocl;opencv_nonfree;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_stitching248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_stitching )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_stitching "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_stitching248.a" )

# Import target "opencv_superres" for configuration "Release"
set_property(TARGET opencv_superres APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_superres PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;opencv_flann;opencv_imgproc;opencv_highgui;opencv_features2d;opencv_calib3d;opencv_ml;opencv_video;opencv_legacy;opencv_objdetect;opencv_photo;opencv_gpu;opencv_ocl;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_superres248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_superres )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_superres "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_superres248.a" )

# Import target "opencv_ts" for configuration "Release"
set_property(TARGET opencv_ts APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_ts PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;opencv_flann;opencv_imgproc;opencv_highgui;opencv_features2d;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_ts248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_ts )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_ts "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_ts248.a" )

# Import target "opencv_videostab" for configuration "Release"
set_property(TARGET opencv_videostab APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_videostab PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "opencv_core;opencv_flann;opencv_imgproc;opencv_highgui;opencv_features2d;opencv_calib3d;opencv_ml;opencv_video;opencv_legacy;opencv_objdetect;opencv_photo;opencv_gpu;stdc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_videostab248.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_videostab )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_videostab "${_IMPORT_PREFIX}/x64/mingw/staticlib/libopencv_videostab248.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
