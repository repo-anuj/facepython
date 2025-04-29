import os
import sys

print(f"Current working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

# Check for emotion_dataset directory
if os.path.exists('emotion_dataset'):
    print("emotion_dataset directory exists")
    folders = [d for d in os.listdir('emotion_dataset') if os.path.isdir(os.path.join('emotion_dataset', d))]
    if folders:
        print(f"Found emotion dataset folders: {', '.join(folders)}")
        
        # Check for image files in each folder
        for folder in folders:
            folder_path = os.path.join('emotion_dataset', folder)
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Folder {folder}: {len(image_files)} image files")
            
            # Print a few example image paths
            if image_files:
                for i, img in enumerate(image_files[:3]):
                    img_path = os.path.join(folder_path, img)
                    print(f"  Example {i+1}: {img_path} (exists: {os.path.exists(img_path)})")
    else:
        print("No emotion dataset folders found")
else:
    print("emotion_dataset directory does not exist")

# Check for models directory
if os.path.exists('models'):
    print("models directory exists")
    files = os.listdir('models')
    if files:
        print(f"Found files in models directory: {', '.join(files)}")
    else:
        print("No files found in models directory")
else:
    print("models directory does not exist")
