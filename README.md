# SDM and Head Pose Estimation

### example

git clone [https://github.com/RoboPai/sdm.git](https://github.com/RoboPai/sdm.git)

    <span class="hljs-keyword">cd</span> sdm/example/
    ./Roboman-SDM-TEST.<span class="hljs-keyword">exe</span>
    `</pre>

    You will look like as follows:

    ![image](https://github.com/RoboPai/sdm/example/1.jpg)

    ![image](https://github.com/RoboPai/sdm/example/2.jpg)

    ![image](https://github.com/RoboPai/sdm/example/3.jpg)

    ![image](https://github.com/RoboPai/sdm/example/4.jpg)

    ### How to compile on windows？

    Make sure you have installed the cmake-gui & Qt5.3(mingw will be needed).

    <pre>`<span class="hljs-built_in">mkdir</span> build
    <span class="hljs-keyword">cd</span> build
    cmake-<span class="hljs-keyword">gui</span>
    mingw32-<span class="hljs-keyword">make</span>
    `</pre>

    ### How to compile on Ubuntu？

    Maybe you should modify the CMakeList.txt about OpenCV

    <pre>`<span class="hljs-built_in">mkdir</span> build
    <span class="hljs-keyword">cd</span> build
    cmake ..
    <span class="hljs-keyword">make</span>
    <span class="hljs-keyword">cp</span> Roboman-SDM-TEST ../example/
    <span class="hljs-keyword">cd</span> ../example/
    ./Roboman-SDM-TEST

### How to compile on iOS？

Add the OpenCV2.framework &amp; src folder to you source codes, compile &amp; enjoy it.

### How to compile on Android？

According to Android cmake