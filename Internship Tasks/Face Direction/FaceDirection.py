import cv2
import mediapipe as mp
import numpy as np
import time

# Create mediapipe modules and implement classes from it.
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh( 
                               max_num_faces=2,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5  )

# Create instances of mediapipe drawing utilities.
mpDrawing = mp.solutions.drawing_utils
drawingSpec = mpDrawing.DrawingSpec((20,120,50),
                                    thickness=1,
                                    circle_radius=1 )

# Read feed from webcam and resize
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

while True:
    isReadOk, frame = cap.read()
    startTime = time.time()
    
    if not(isReadOk):
        print("could'nt read the feed from webcam.")
        break
    
    # Get frames dimentions
    frameH, frameW, frameCh = frame.shape
    
    # Create 3D and 2D list of informations.
    face3D = []
    face2D = []
    
    # Edit input frame to be RGB
    # and flip it to be as a selfie.
    frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
    # make it not writeable to procces faster.
    frame.flags.writeable = False
    
    # Process the frame.
    result = faceMesh.process(frame)
    
    # make it writeable.
    frame.flags.writeable = True
    
    # Convert it back to BGR to be able to work on it.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Check if results have been aquierd then:
    if result.multi_face_landmarks:
        # Loop trough landmarks.
        for faceLandmarks in result.multi_face_landmarks:
            # Loop trough landmarks details
            for idx, landmark in enumerate(faceLandmarks.landmark):
                #print(f'idx:{idx} | landmark:{landmark}')
                # Check for a valuable point in landmarks:
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose2D = (landmark.x * frameW, landmark.y * frameH)
                        nose3D = (landmark.x * frameW, landmark.y * frameH, landmark.z * 100)
            
                x, y = int(landmark.x * frameW), int(landmark.y * frameH)
                
                # 2D coordinations
                face2D.append([x, y])
                
                # 3D coordinations
                face3D.append([x, y, landmark.z])
                
            # Convert to numpy array:
            face2D = np.array(face2D, dtype=np.float64)
            face3D = np.array(face3D, dtype=np.float64)
            
            # Camera matrice
            focalLength = frameW * 1
            cameraMatrix = np.array([[focalLength,  0,              frameH/2],
                                     [0,            focalLength,    frameW/2],
                                     [0,            0,              1       ]] )
            
            # Distortion
            distortionMatrix = np.zeros((4, 1), dtype=np.float64)
            
            # -------- Perspective-n-Point ----------
            success, rotationVector, translationVector =\
            cv2.solvePnP(face3D, face2D, cameraMatrix, distortionMatrix)
            
            # Get rotational matrix
            rotationalMatrix, jac = cv2.Rodrigues(rotationVector)
            
            # Get angles
            angles, ntxr, mtxq, qx,qy,qz = cv2.RQDecomp3x3(rotationalMatrix)
            
            # Get rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
            
            # Display tilting
            # pass
            
            # Display nose direction
            nose3DProjection, jacobian = cv2.projectPoints(nose3D,
                                                           rotationVector,
                                                           translationVector,
                                                           cameraMatrix,
                                                           distortionMatrix )
            
            p1 = (int(nose2D[0]), int(nose2D[1]))
            p2 = (int(nose3DProjection[0][0][0] + y*30 ) , int(nose3DProjection[0][0][1]- x*20))
            
            cv2.line(frame, p1, p2, (255, 255, 255), 8)
            
            # Draw face mesh as TESSELATION attribiute.
            mpDrawing.draw_landmarks(
                                     frame,
                                     faceLandmarks,
                                     mpFaceMesh.FACEMESH_TESSELATION,
                                     drawingSpec,
                                     drawingSpec    )
    endTime = time.time()
    fps = 1 / (endTime-startTime)
    cv2.putText(frame, f"fps:{int(fps)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (20,20,20), 2)
    cv2.imshow("Capture", frame)
    
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break