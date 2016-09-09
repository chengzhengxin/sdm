# SDM and Head Pose Estimation

### example

git clone [https://github.com/RoboPai/sdm.git](https://github.com/RoboPai/sdm.git)

	cd sdm/example/
    ./Roboman-SDM-TEST.exe

You will look like as follows:

![image](https://github.com/RoboPai/sdm/raw/master/example/1.jpg)

![image](https://github.com/RoboPai/sdm/raw/master/example/2.jpg)

![image](https://github.com/RoboPai/sdm/raw/master/example/3.jpg)

![image](https://github.com/RoboPai/sdm/raw/master/example/4.jpg)

### How to compile on Windows？

Make sure you have installed the cmake-gui & Qt5.3.2 (mingw will be needed).

	mkdir build
	cd build
	cmake-gui
	mingw32-make

### How to compile on Ubuntu？

Maybe you should modify the CMakeLists.txt about OpenCV libs

	mkdir build
	cd build
	cmake ..
	make
	cp Roboman-SDM-TEST ../example/
	cd ../example/
	./Roboman-SDM-TEST

### How to compile on iOS？

Add the OpenCV2.framework & src folder to you source codes, compile & enjoy it.

### How to compile on Android？

According to Android-cmake-Tool.