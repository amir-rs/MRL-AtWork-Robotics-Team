12/04/1401

# Problem:
At this time agent doesn't calculate coordinates to put down objects, on the platform.


# Solution

Goal(s):

1.  Have a FOV of the platform (which, objects are going to be put on),
    the robot should place objects according to their dimensions and the platform's available areas.
    
    STATE: developing
    
    # try 1: 1401/04 - 1401/06
        Approach:
            using basic threshold and color segmentation.
        
        [*]  1. choosing preprocessing algorithms [blur,...]
            [cv.THRESH_BINARY + cv.THRESH_OTSU] was not good enough.
            [cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY] has far better result.
            using [cv.morphologyEx(threshold_frame, cv.MORPH_GRADIENT, kernel)] gives more accurate results.

        

        Result:
            In the end, it's not accurate and the results are not good enough.
    ------------------------------
    
    # try 2: 1401/11 - 1401/12
        Approach:
            using OpenCV's background subtraction and gaussian mixture algorithms.
                    
        [*]  KNN: 7/10.
        [*]  GSOC: 5/10
        [*]  DOG:
        [ ]  MOG2:

        Result:
            results are not acceptable.
            better than the last approach but has some bug and fails at some point when there is environmental effects such as,
            light rays or platform material texture(like grass or wood texture).

            *DOG: needs tweaks on parameters, result is already competitive.
    ------------------------------
    
    # try 3: 1401/12 - ???
        Approach:
            using depth data to find objects.

        []  1. Analyze object detection methods on depth data.

        Result:
            ???
    ------------------------------
    
    # try 4: 1401/12 - ???
        Approach:
            create a custom lightweight object detection tool. 

        Result:
            ???
    ------------------------------


        
        
2.  Have an algorithm to arrange objects in such a way,
    to be the optimum state of placing objects according to proportions.(like a puzzle)
    
    STATE: comming

note:
    1. all functions have a document part (written as comments) above them that exhibit what's their task and how they get it done.
    2. documentation structure:
        TAKES(function inputs) [...]
        DOES(what this function does) [...]
        RETURN(function output) [...]

        each part notation is represented in this way:
        x-ly_lz         ---->       [x]: task number,
                                    [y]: beginning line,
                                    [z]: finishing line.