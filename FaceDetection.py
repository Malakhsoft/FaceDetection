import cv2
import os

print("███╗░░░███╗░█████╗░██╗░░░░░░█████╗░██╗░░██╗██╗░░██╗")
print("████╗░████║██╔══██╗██║░░░░░██╔══██╗██║░██╔╝██║░░██║")
print("██╔████╔██║███████║██║░░░░░███████║█████═╝░███████║")
print("██║╚██╔╝██║██╔══██║██║░░░░░██╔══██║██╔═██╗░██╔══██║")
print("██║░╚═╝░██║██║░░██║███████╗██║░░██║██║░╚██╗██║░░██║")
print("╚═╝░░░░░╚═╝╚═╝░░╚═╝╚══════╝╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚═╝\n")

print("░██████╗░█████╗░███████╗████████╗░██╗░░░░░░░██╗░█████╗░██████╗░███████╗░██████╗")
print("██╔════╝██╔══██╗██╔════╝╚══██╔══╝░██║░░██╗░░██║██╔══██╗██╔══██╗██╔════╝██╔════╝")
print("╚█████╗░██║░░██║█████╗░░░░░██║░░░░╚██╗████╗██╔╝███████║██████╔╝█████╗░░╚█████╗░")
print("░╚═══██╗██║░░██║██╔══╝░░░░░██║░░░░░████╔═████║░██╔══██║██╔══██╗██╔══╝░░░╚═══██╗")
print("██████╔╝╚█████╔╝██║░░░░░░░░██║░░░░░╚██╔╝░╚██╔╝░██║░░██║██║░░██║███████╗██████╔╝")
print("╚═════╝░░╚════╝░╚═╝░░░░░░░░╚═╝░░░░░░╚═╝░░░╚═╝░░╚═╝░░╚═╝╚═╝░░╚═╝╚══════╝╚═════╝\n")


def display_gpl():
    gpl_text = """
    GNU GENERAL PUBLIC LICENSE
    Version 3, 29 June 2007

    Copyright (C) Malakh Softwares. 

    Everyone is permitted to copy and distribute verbatim or modified
    copies of this program and its license documentations, with or
    without modifications, provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions, and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions, and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

    3. Neither the name of Malakh nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    """

    print(gpl_text)

# Call the function to display the GPL
display_gpl()


def detect_faces(frame):
    # Load the pre-trained face cascade from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw green squares around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame, faces

def main():
    # Get the number of available webcams
    num_cameras = 2  # You may need to adjust this based on the number of cameras on your computer

    # Open a window for each camera
    cap_list = [cv2.VideoCapture(i) for i in range(num_cameras)]

    # Create a directory to store video clips
    clips_dir = 'clips'
    os.makedirs(clips_dir, exist_ok=True)

    # Define video writer parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer_list = [None] * num_cameras

    while True:
        face_count = 0  # Reset face count for each iteration

        # Capture frames from each camera
        frames = [cap.read()[1] for cap in cap_list]

        # Detect faces and draw green squares for each frame
        frames_with_faces = [detect_faces(frame) for frame in frames]

        # Display frames in separate windows with face count
        for i, (frame_with_faces, faces) in enumerate(frames_with_faces):
            # Display face count in the top-left corner
            cv2.putText(frame_with_faces, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Display "Press Q to exit" in red at the bottom right corner
            cv2.putText(frame_with_faces, 'Press Q to exit', (frame_with_faces.shape[1]-170, frame_with_faces.shape[0]-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            # Display the program message in gray at the bottom center
            program_message = 'This program is Free of Charge and Designed by Matt Mirzaei'
            text_size = cv2.getTextSize(program_message, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = (frame_with_faces.shape[1] - text_size[0]) // 2
            text_y = frame_with_faces.shape[0] - 10
            cv2.putText(frame_with_faces, program_message, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2, cv2.LINE_AA)

            cv2.imshow(f'Camera {i+1} - Face Detection', frame_with_faces)

            # Record video clips only when a face is detected
            if len(faces) > 0:
                if video_writer_list[i] is None:
                    video_writer_list[i] = cv2.VideoWriter(os.path.join(clips_dir, f'camera_{i+1}_clip.mp4'), fourcc, 20.0, (640, 480))
                video_writer_list[i].write(frame_with_faces)
                #-Debug-print(f'Recording: camera_{i+1}_clip.mp4')

            # Update total face count
            face_count += len(faces)

        # Display total face count in the console
        #-Debug-print(f'Total Faces: {face_count}')

        # Check for the 'Q' key to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcams, video writers, and close the OpenCV windows
    for cap, video_writer in zip(cap_list, video_writer_list):
        cap.release()
        if video_writer is not None:
            video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
