import torch
import torchvision.models as models
from torchvision import transforms as T
from PIL import Image
import cv2
from mtcnn import MTCNN
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import clip  # Import CLIP

# Define the emotion model architecture using ResNet-50 instead of EfficientNet
model = models.resnet50(weights=None)
# Modify first conv layer to accept grayscale input (1 channel)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# Change final layer to output 10 classes
model.fc = torch.nn.Linear(model.fc.in_features, 8)  # 10 emotions

# Load the trained weights
model.load_state_dict(torch.load("best (1).pth", map_location=torch.device('cpu')))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load CLIP model
print("Loading CLIP model...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print("CLIP model loaded successfully")

# Use YOLOv5 instead since YOLOv9 may not be directly available via torch.hub
try:
    # Try to load YOLOv9 first
    yolo_model = torch.hub.load('ultralytics/yolov9', 'yolov9s', pretrained=True)
except Exception as e:
    print(f"Couldn't load YOLOv9, falling back to YOLOv5: {e}")
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

yolo_model.to(device)
yolo_model.eval()

# Define transforms for image preprocessing
val_tfm = T.Compose([
    T.Grayscale(1),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

# Define emotion labels
emotion_labels = ["neutral", "happy", "surprise", "sad", "anger", "disgust", "fear", "contempt"]

# Define objects that might influence emotion context
relevant_objects = ['cake', 'balloon', 'gift', 'dog', 'cat', 'flower', 
                    'hospital_bed', 'gun', 'knife', 'car', 'person', 
                    'book', 'tv', 'laptop', 'phone', 'bottle', 'fire']  # Added 'fire'

# Scene categories and emotional contexts for CLIP analysis
scene_categories = [
    "birthday party", "office meeting", "funeral", "classroom", "outdoor park", 
    "restaurant", "hospital room", "living room", "concert", "sports event",
    "wedding", "graduation ceremony", "beach scene", "forest", "nightclub",
    "war zone", "crime scene", "peaceful garden"
]

emotional_contexts = [
    "happy occasion", "sad event", "tense situation", "peaceful setting", 
    "exciting moment", "romantic scene", "scary environment", "neutral setting",
    "celebratory atmosphere", "serious business environment", "relaxing environment",
    "dangerous situation", "playful setting"
]

def analyze_scene_with_clip(frame):
    """Analyze scene type and emotional context using CLIP model"""
    # Convert frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Preprocess for CLIP
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    
    # Analyze scene category
    text_inputs_scene = torch.cat([clip.tokenize(f"a photo of {c}") for c in scene_categories]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs_scene)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    # Get top scene
    values, indices = similarity[0].topk(1)
    scene_type = scene_categories[indices[0].item()]
    scene_confidence = values[0].item()
    
    # Analyze emotional context
    text_inputs_emotion = torch.cat([clip.tokenize(f"a photo of {e}") for e in emotional_contexts]).to(device) 
    with torch.no_grad():
        emotion_features = clip_model.encode_text(text_inputs_emotion)
        emotion_features /= emotion_features.norm(dim=-1, keepdim=True)
        emotion_similarity = (100.0 * image_features @ emotion_features.T).softmax(dim=-1)
    
    # Get top emotional context
    e_values, e_indices = emotion_similarity[0].topk(1)
    emotional_context = emotional_contexts[e_indices[0].item()]
    emotional_confidence = e_values[0].item()
    
    return {
        "scene_type": scene_type,
        "scene_confidence": scene_confidence,
        "emotional_context": emotional_context,
        "emotional_confidence": emotional_confidence
    }

class EmotionRefiner:
    def __init__(self):
        # Define emotional valence of objects (statistical associations)
        # Format: object_name: {emotion: weight}
        self.object_emotion_weights = {
            'cake': {'happy': 0.6, 'neutral': 0.2},
            'balloon': {'happy': 0.5, 'surprise': 0.3},
            'gift': {'happy': 0.7, 'surprise': 0.4},
            'dog': {'happy': 0.5, 'neutral': 0.2},
            'cat': {'happy': 0.4, 'neutral': 0.3},
            'flower': {'happy': 0.4},
            'hospital_bed': {'sad': 0.7, 'neutral': 0.1},
            'gun': {'fear': 0.7, 'anger': 0.2},  # Changed from 'angry' to 'anger'
            'knife': {'fear': 0.6, 'anger': 0.3},  # Changed from 'angry' to 'anger'
            'car': {'neutral': 0.5},
            'person': {'neutral': 0.2},  # Added default weight
            'book': {'neutral': 0.3},
            'tv': {'neutral': 0.3},
            'laptop': {'neutral': 0.3},
            'phone': {'neutral': 0.3},
            'bottle': {'neutral': 0.2},
            'fire': {'fear': 0.5, 'surprise': 0.3},
        }
        
        # Define scene-to-emotion associations
        self.scene_emotion_map = {
            "birthday party": {"happy": 0.8, "surprise": 0.5},
            "wedding": {"happy": 0.8, "neutral": 0.2},
            "graduation ceremony": {"happy": 0.7, "neutral": 0.3},
            "funeral": {"sad": 0.9, "neutral": 0.4},
            "hospital room": {"sad": 0.7, "fear": 0.4},
            "classroom": {"neutral": 0.6, "happy": 0.2},
            "office meeting": {"neutral": 0.7},
            "outdoor park": {"happy": 0.5, "neutral": 0.4},
            "beach scene": {"happy": 0.6, "neutral": 0.3},
            "forest": {"neutral": 0.5, "happy": 0.4},
            "nightclub": {"happy": 0.5, "surprise": 0.4},
            "war zone": {"fear": 0.8, "sad": 0.7},
            "crime scene": {"fear": 0.7, "sad": 0.6},
            "peaceful garden": {"neutral": 0.7, "happy": 0.3},
        }
        
        # Define emotional context associations
        self.context_emotion_map = {
            "happy occasion": {"happy": 0.8, "surprise": 0.4},
            "sad event": {"sad": 0.8, "neutral": 0.3},
            "tense situation": {"fear": 0.7, "anger": 0.5},  # Changed from 'angry' to 'anger'
            "peaceful setting": {"neutral": 0.6, "happy": 0.3},
            "exciting moment": {"surprise": 0.7, "happy": 0.5},
            "romantic scene": {"happy": 0.6},
            "scary environment": {"fear": 0.8, "surprise": 0.4},
            "neutral setting": {"neutral": 0.8},
            "celebratory atmosphere": {"happy": 0.8, "surprise": 0.5},
            "serious business environment": {"neutral": 0.7},
            "relaxing environment": {"neutral": 0.6, "happy": 0.2},
            "dangerous situation": {"fear": 0.8, "anger": 0.4},  # Changed from 'angry' to 'anger'
            "playful setting": {"happy": 0.7, "surprise": 0.3},
        }
        
        # Add emotional associations for contempt, unknown, and NF
        self._add_additional_emotions()
        
        # Base confidence required to maintain original prediction
        self.base_confidence = 70.0
        
    def _add_additional_emotions(self):
        """Add support for contempt, unknown and NF emotions"""
        # Add contempt associations to objects
        for obj in ['knife', 'gun', 'person']:
            if obj in self.object_emotion_weights:
                self.object_emotion_weights[obj]['contempt'] = 0.3
        
        # Add contempt to scenes
        for scene in ["crime scene", "war zone", "office meeting"]:
            if scene in self.scene_emotion_map:
                self.scene_emotion_map[scene]['contempt'] = 0.4
        
        # Add contempt to contexts
        for context in ["tense situation", "serious business environment"]:
            if context in self.context_emotion_map:
                self.context_emotion_map[context]['contempt'] = 0.4

    def refine_emotion_with_clip(self, emotion, objects, confidence, scene_analysis):
        """Enhanced method using CLIP scene analysis"""
        # If confidence is very high, trust the original prediction
        if confidence > 90.0:
            return emotion
            
        # Calculate emotional influence of detected objects
        emotion_scores = {e: 0.0 for e in emotion_labels}
        
        # Add base value for detected emotion
        emotion_scores[emotion] = max(0, (confidence - self.base_confidence) / 100.0)
        
        # Calculate influence from each object
        for obj in objects:
            if obj in self.object_emotion_weights:
                # Add emotional influence from this object
                for emotion_name, weight in self.object_emotion_weights[obj].items():
                    if emotion_name in emotion_scores:  # Make sure emotion exists in labels
                        emotion_scores[emotion_name] += weight
        
        # Add scene type influence with stronger weight
        scene_type = scene_analysis["scene_type"]
        scene_conf = scene_analysis["scene_confidence"]
        
        if scene_type in self.scene_emotion_map:
            for emotion_name, weight in self.scene_emotion_map[scene_type].items():
                if emotion_name in emotion_scores:  # Make sure emotion exists in labels
                    emotion_scores[emotion_name] += weight * scene_conf
        
        # Add emotional context influence
        emotional_context = scene_analysis["emotional_context"]
        emotional_conf = scene_analysis["emotional_confidence"]
        
        if emotional_context in self.context_emotion_map:
            for emotion_name, weight in self.context_emotion_map[emotional_context].items():
                if emotion_name in emotion_scores:  # Make sure emotion exists in labels
                    emotion_scores[emotion_name] += weight * emotional_conf
        
        # Get emotion with highest total score
        max_emotion = max(emotion_scores, key=emotion_scores.get)
        max_score = emotion_scores[max_emotion]
        
        # Only refine if the score is significant enough
        if max_emotion != emotion and max_score > emotion_scores[emotion] + 0.3:
            return max_emotion
        
        return emotion

# Initialize the face detector and emotion refiner
detector = MTCNN()
refiner = EmotionRefiner()

# Detect objects in the scene
def detect_objects(frame):
    """
    Detect objects in the frame using the YOLO model.
    Returns a list of detected object names and their bounding boxes.
    """
    # Run the model on the frame
    results = yolo_model(frame)
    
    # Extract results
    detected_objects = []
    object_boxes = []
    
    # Process detections
    for detection in results.xyxy[0]:  # results.xyxy[0] contains detections for the first (and only) image
        # Get detection info (coordinates, confidence, class)
        x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
        
        # Convert coordinates and confidence to integers/float
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = float(conf)
        
        # Get class name
        class_name = results.names[int(cls)]
        
        # Only include relevant objects or objects with high confidence
        if class_name in relevant_objects or conf > 0.7:
            detected_objects.append(class_name)
            object_boxes.append({
                'box': (x1, y1, x2, y2),
                'conf': conf,
                'class': class_name
            })
    
    return detected_objects, object_boxes


# Generate a distinct color for each class
def get_color(class_name):
    """Generate a distinct color for any YOLO class."""
    # Custom colors for our specifically relevant objects
    custom_colors = {
        'cake': (255, 0, 0),      # Red
        'balloon': (0, 255, 0),   # Green
        'gift': (0, 0, 255),      # Blue
        'dog': (255, 255, 0),     # Yellow
        'cat': (255, 0, 255),     # Magenta
        'flower': (0, 255, 255),  # Cyan
        'hospital_bed': (128, 0, 0),  # Dark red
        'gun': (0, 128, 0),       # Dark green
        'knife': (0, 0, 128),     # Dark blue
        'car': (128, 128, 0),     # Olive
        'person': (128, 0, 128),  # Purple
        'book': (0, 128, 128),    # Teal
        'tv': (128, 128, 128),    # Gray
        'laptop': (64, 0, 0),     # Brown
        'phone': (0, 64, 0),      # Dark green
        'bottle': (0, 0, 64),     # Navy
        'fire': (255, 140, 0)     # Orange
    }
    
    # Return custom color if defined
    if class_name in custom_colors:
        return custom_colors[class_name]
    
    # Generate deterministic color based on class name for any other class
    # This ensures each class gets a unique but consistent color
    hash_val = hash(class_name) & 0xFFFFFF
    r = hash_val & 0xFF
    g = (hash_val >> 8) & 0xFF
    b = (hash_val >> 16) & 0xFF
    
    # Ensure color is bright enough to be visible
    min_brightness = 100
    r = max(r, min_brightness)
    g = max(g, min_brightness)
    b = max(b, min_brightness)
    
    return (int(r), int(g), int(b))

# Start webcam processing
def main():
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    
    # Frame counter for CLIP analysis (to avoid running on every frame)
    frame_count = 0
    clip_interval = 10  # Analyze scene every 10 frames
    last_scene_analysis = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Analyze scene with CLIP periodically to save resources
        if frame_count % clip_interval == 0:
            try:
                scene_analysis = analyze_scene_with_clip(frame)
                last_scene_analysis = scene_analysis
                print(f"Scene analysis: {scene_analysis}")
            except Exception as e:
                print(f"CLIP analysis error: {str(e)}")
                scene_analysis = last_scene_analysis  # Use previous analysis if error
        else:
            scene_analysis = last_scene_analysis
            
        frame_count += 1
        
        detected_objects, object_boxes = detect_objects(frame)
        
        # Display detected objects
        objects_text = ", ".join(detected_objects) if detected_objects else "None"
        cv2.putText(frame, f"Objects: {objects_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
        # Display scene context information if available
        if scene_analysis:
            scene_text = f"Scene: {scene_analysis['scene_type']} ({scene_analysis['scene_confidence']:.2f})"
            mood_text = f"Mood: {scene_analysis['emotional_context']} ({scene_analysis['emotional_confidence']:.2f})"
            
            cv2.putText(frame, scene_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.putText(frame, mood_text, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        
        # Draw bounding boxes around detected objects
        for obj in object_boxes:
            x1, y1, x2, y2 = obj['box']
            class_name = obj['class']
            confidence = obj['conf']
            
            # Get color for this class
            color = get_color(class_name)
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Detect faces
        try:
            faces = detector.detect_faces(frame)
            for face in faces:
                x, y, w, h = face['box']
                
                # Ensure box coordinates are valid
                x, y = max(0, x), max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                
                # Skip if face area is too small
                if w <= 0 or h <= 0:
                    continue
                    
                face_img = frame[y:y+h, x:x+w]
                
                # Only process if face area is valid
                if face_img.size > 0:
                    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).convert("L")
                    face_tensor = val_tfm(face_pil).unsqueeze(0).to(device)
                    
                    # Predict emotion
                    with torch.no_grad():
                        logits = model(face_tensor)
                        probs = F.softmax(logits, dim=1)
                        confidence, pred = torch.max(probs, 1)
                        pred = pred.item()
                        confidence = confidence.item() * 100
                    
                    initial_emotion = emotion_labels[pred]
                    
                    # Refine emotion based on scene context and objects
                    if scene_analysis:
                        refined_emotion = refiner.refine_emotion_with_clip(
                            initial_emotion, 
                            detected_objects, 
                            confidence, 
                            scene_analysis
                        )
                    else:
                        # Fallback to object-only refinement
                        refined_emotion = refiner.refine_emotion(
                            initial_emotion, 
                            detected_objects, 
                            confidence
                        )
                    
                    # Display results
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display initial emotion with confidence
                    initial_label = f"Initial: {initial_emotion}: {confidence:.2f}%"
                    cv2.putText(frame, initial_label, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display refined emotion
                    refined_label = f"Refined: {refined_emotion}"
                    cv2.putText(frame, refined_label, (x, y-35), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        except Exception as e:
            # Add error handling for face detection
            cv2.putText(frame, f"Face detection error: {str(e)}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Emotion Detection with Scene Understanding', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()