import os
import cv2
import time
import numpy as np
from datetime import datetime

class DataCollector:
    def __init__(self, base_dir='collected_data'):
        """Initialize the data collector with a base directory for storing images."""
        self.base_dir = base_dir
        self.current_person = None
        self.collection_active = False
        self.collection_count = 0
        self.max_images = 50  # Default maximum images to collect per person
        self.collection_interval = 0.5  # Time between captures in seconds
        self.last_capture_time = 0
        
        # Create base directory if it doesn't exist
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
    
    def start_collection(self, person_name, max_images=50):
        """Start collecting face images for a person."""
        if not person_name or person_name.strip() == "":
            return False, "Person name cannot be empty"
        
        # Sanitize person name for folder name (remove special characters)
        person_name = ''.join(c for c in person_name if c.isalnum() or c in [' ', '_']).strip()
        person_name = person_name.replace(' ', '_')
        
        # Create directory for this person
        person_dir = os.path.join(self.base_dir, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        self.current_person = person_name
        self.collection_active = True
        self.collection_count = 0
        self.max_images = max_images
        self.last_capture_time = 0
        
        return True, f"Started collection for {person_name}"
    
    def stop_collection(self):
        """Stop the current collection process."""
        if not self.collection_active:
            return False, "No active collection to stop"
        
        result = f"Stopped collection for {self.current_person}. Collected {self.collection_count} images."
        self.collection_active = False
        self.current_person = None
        
        return True, result
    
    def collect_face(self, frame, face_coords):
        """
        Collect a face image if collection is active.
        
        Args:
            frame: The full camera frame
            face_coords: Tuple of (x, y, w, h) coordinates of the face
        
        Returns:
            (success, message): Tuple indicating success and a message
        """
        if not self.collection_active:
            return False, "No active collection"
        
        # Check if we've reached the maximum number of images
        if self.collection_count >= self.max_images:
            self.collection_active = False
            return False, f"Collection complete for {self.current_person}. Maximum images reached."
        
        # Check if enough time has passed since the last capture
        current_time = time.time()
        if current_time - self.last_capture_time < self.collection_interval:
            return False, "Too soon for next capture"
        
        # Extract face from frame
        x, y, w, h = face_coords
        face_img = frame[y:y+h, x:x+w]
        
        # Resize to a standard size (e.g., 224x224 for VGG16)
        face_img = cv2.resize(face_img, (224, 224))
        
        # Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.current_person}_{timestamp}.jpg"
        filepath = os.path.join(self.base_dir, self.current_person, filename)
        
        # Save the image
        cv2.imwrite(filepath, face_img)
        
        # Update counters
        self.collection_count += 1
        self.last_capture_time = current_time
        
        return True, f"Captured image {self.collection_count}/{self.max_images} for {self.current_person}"
    
    def get_collection_status(self):
        """Get the current status of data collection."""
        if not self.collection_active:
            return {
                'active': False,
                'person': None,
                'count': 0,
                'max': 0,
                'progress': 0
            }
        
        progress = int((self.collection_count / self.max_images) * 100)
        return {
            'active': True,
            'person': self.current_person,
            'count': self.collection_count,
            'max': self.max_images,
            'progress': progress
        }
    
    def get_collected_people(self):
        """Get a list of people for whom data has been collected."""
        if not os.path.exists(self.base_dir):
            return []
        
        people = []
        for person_dir in os.listdir(self.base_dir):
            dir_path = os.path.join(self.base_dir, person_dir)
            if os.path.isdir(dir_path):
                # Count the number of images
                image_count = len([f for f in os.listdir(dir_path) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                people.append({
                    'name': person_dir,
                    'image_count': image_count
                })
        
        return people
