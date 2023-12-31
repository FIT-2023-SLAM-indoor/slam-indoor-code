# slam-indoor-code
Repository with source code of SLAM indoor project

## Requirements
- OpenCV 4.8.0 
- CMake 3.26.2 ([release](https://github.com/Kitware/CMake/releases/tag/v3.27.6))

## Environment configuration
1. Download release [here](https://github.com/opencv/opencv/releases/tag/4.8.0) (**For Windows ONLY `.exe`**)
2. Extract files somewhere (e.g. `C:\`)
3. Add folders `bin` and `lib` to path (e.g. for windows x64: `opencv\build\x64\vc16\bin` and `opencv\build\x64\vc16\lib`)
4. *For VS Code*:
   1. Install extension [C/C++ Extension pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack)
   2. Install extension [CMake](https://marketplace.visualstudio.com/items?itemName=twxs.cmake)
   3. Press `F1` then type `CMake: configure`
5. *For MS VS*:
   1. Press `Project -> Configure project with CMake debugging`
   2. Right-click on `CMakeLists.txt` then press "Open debug and launch settings" and add to json `"currentDir": "${workspaceRoot}"`
   3. Launch code using startup item `CMakeLists.txt`