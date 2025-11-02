Automatic Traffic signal Control System (ATSC) with Emergency Vehicle Detection

This an AI-powered solution that uses object detection (YOLOv8) to manage urban traffic more efficiently and sustainably. The system analyzes real-time traffic footage to determine vehicle density and detect emergency vehicles such as ambulances and fire trucks. Based on these insights, it dynamically adjusts traffic light durations, ensuring smoother flow, quicker emergency response, and reduced emissions.


Features
Real-Time Vehicle Detection: Identifies and counts cars, bikes, buses, and trucks.
Adaptive Signal Control: Adjusts green light duration based on real-time traffic density.
Emergency Vehicle Priority: Detects emergency vehicles and prioritizes their lane.
Eco-Friendly Design: Minimizes idle time and reduces CO₂ emissions.


Tech Stack

Programming Language :Python 3.10+
AI Model : YOLOv8 (Ultralytics)
Computer Vision : OpenCV
Data Annotation : LabelImg / Roboflow Annotate

System Workflow
1. Input: Real-time or recorded traffic video.
2. Detection: YOLOv8 identifies vehicle types and detects emergency vehicles.
3. Analysis: Counts vehicles in each lane to determine density.
4. Signal Control: Adjusts traffic light duration or provides emergency clearance.
5. Output: Updated traffic light sequence optimizing flow and reducing congestion.