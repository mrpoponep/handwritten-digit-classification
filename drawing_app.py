import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Global variables
drawing = False
ix, iy = -1, -1

class Number_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=15,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=15,out_channels=10,kernel_size=3,stride=1,padding=1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=10*28*28, 
                      out_features=10)
        )
    def forward(self,x):
        x = self.layer(x)   
        x = self.classifier(x)
        return x
#load img and trained model
loaded_model = Number_classifier()
loaded_model.load_state_dict(torch.load("models/03_numberclassifier.pth"))
image_path = 'drawing.png'

transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Để đảm bảo kích thước ảnh là 28x28
    transforms.Grayscale(num_output_channels=1),  # Chuyển ảnh màu về ảnh đen trắng
    transforms.ToTensor(),  # Chuyển ảnh về tensor
    transforms.Normalize((0.5,), (0.5,))  # Chuẩn hóa giá trị pixel về [-1, 1]
])

def draw(event, x, y, flags, param):
    
    global ix, iy, drawing, mode
    if event == cv2.EVENT_LBUTTONDOWN and y<280:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and y<280: 
        if drawing:
            cv2.circle(img, (x, y), 10, (255, 255, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP and y<280:
        drawing = False
        cv2.circle(img, (x, y), 5, (255, 255, 255), -1)

def main():
    global img

    img = np.zeros((380, 280, 3), np.uint8) * 255  # White canvas
    cv2.namedWindow('Drawing App')
    cv2.setMouseCallback('Drawing App', draw)
    cv2.line(img,(0,280),(280,280),(255,2555,255),5)
    while True:
        cv2.imshow('Drawing App', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            img.fill(0)
            cv2.line(img,(0,280),(280,280),(255,2555,255),5)
        elif key == 32: #Space key
            cv2.imwrite('drawing.png', img)
            image = Image.open(image_path)
            image=transforms.functional.crop(image,0,0,280,280)
            image_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = loaded_model(image_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                print("Predicted digit:", predicted_class)
                cv2.putText(img,f"{predicted_class}",(130,340),5,2,(255,255,255))
        elif key == 27:  # Esc key
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
