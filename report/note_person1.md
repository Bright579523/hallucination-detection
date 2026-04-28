# NOTE PERSON 1 — DATA ENGINEER
  Prompt Hallucination Detection in Open-Vocabulary Object Detection
# THAI VERSION (ภาษาไทย)
---
## 1. บทบาทและความรับผิดชอบ
---
Person 1 รับบทบาทเป็น Data Engineer ซึ่งเป็นผู้รับผิดชอบในการจัดเตรียมข้อมูล
ทั้งหมดของโปรเจกต์ ตั้งแต่การดาวน์โหลดชุดข้อมูล COCO val2017 ไปจนถึงการออกแบบ
และสร้างชุดข้อมูลสำหรับทดสอบ Hallucination Detection โดยใช้เทคนิค
Supercategory-based Hard Negative Sampling ซึ่งถือเป็นจุดเด่น (Novelty)
ของโปรเจกต์นี้ในระดับ Dataset Design
ไฟล์ที่รับผิดชอบ:
  - data/build_dataset.py     สคริปต์หลักสร้างชุดข้อมูล
  - data/download_coco.py     สคริปต์ดาวน์โหลด COCO
  - data/mockup_dataset.csv   ข้อมูลจำลอง 10 แถว
  - config.py                 จัดการ Path อัตโนมัติ Local/Colab
  - requirements.txt          รายการ Dependencies
  - .gitignore                ป้องกันอัปโหลดไฟล์ขนาดใหญ่
---
## 2. แหล่งข้อมูล (Data Source)
---
โปรเจกต์นี้ใช้ชุดข้อมูล COCO val2017 (Common Objects in Context) เป็นแหล่งข้อมูลหลัก
  ชื่อชุดข้อมูล      : COCO val2017
  จำนวนภาพ          : 5,000 ภาพ
  ประเภท Annotation  : Instance Segmentation (Bounding Box + Category Label)
  จำนวน Category     : 80 ประเภท (เช่น dog, cat, car, airplane, chair)
  จำนวน Supercategory: 12 กลุ่มใหญ่ (เช่น animal, vehicle, furniture, food)
  ไฟล์ Annotation    : instances_val2017.json (รูปแบบ JSON)
  ที่มา             : https://cocodataset.org
เหตุผลที่เลือก COCO val2017:
  - เป็น Benchmark ที่ได้รับการยอมรับในระดับสากลในงาน Object Detection
  - มีโครงสร้าง Supercategory ในตัว ทำให้สามารถสร้าง Hard Negatives ได้
    โดยไม่ต้องใช้ข้อมูลภายนอก
  - ข้อมูลมีคุณภาพสูง Annotation ได้รับการตรวจสอบแล้ว
    ทำให้ไม่ต้องเสียเวลา Data Cleaning
  - ขนาด 5,000 ภาพเหมาะสมกับกรอบเวลาของโปรเจกต์
---
## 3. ขั้นตอนการจัดเตรียมข้อมูล (Data Preparation Pipeline)
---
3.1 การดาวน์โหลดข้อมูล (download_coco.py)
  สคริปต์ download_coco.py ดาวน์โหลดไฟล์ 2 ชุด:
## 1. val2017.zip (~1 GB)                    ภาพถ่าย 5,000 ภาพ
## 2. annotations_trainval2017.zip (~241 MB)  ไฟล์ Annotation
  คุณสมบัติเด่น:
    - รองรับทั้ง Local และ Google Colab ผ่าน config.py
    - มีระบบ Skip อัตโนมัติ ถ้าเคยดาวน์โหลดแล้ว (ไม่โหลดซ้ำ)
    - แตกไฟล์ Zip อัตโนมัติ แล้วลบ Zip ทิ้งเพื่อประหยัดพื้นที่
    - มีขั้นตอน Verification ตรวจสอบจำนวนไฟล์หลังดาวน์โหลด
  คำสั่งรัน:
    python data/download_coco.py                          (ใช้ Default Path)
    python data/download_coco.py --data_dir /content/data  (ระบุ Path เอง)
3.2 การสร้างชุดข้อมูล (build_dataset.py)
  นี่คือหัวใจหลักของงาน Person 1 สคริปต์นี้อ่านไฟล์ instances_val2017.json
  ผ่านไลบรารี pycocotools และสร้างชุดข้อมูล 3 ประเภท:
  ประเภทที่ 1: Positive Samples (label = 0)
    ความหมาย : ภาพที่มีวัตถุนั้นอยู่จริง จับคู่กับ Prompt ที่ถูกต้อง
    ตัวอย่าง : ภาพที่มีสุนัข + prompt "a photo of dog"
    วิธีสร้าง : สุ่มเลือกภาพจาก COCO ที่มี Category ตรงกับ Prompt
    จำนวน    : 2,000 ตัวอย่าง
  ประเภทที่ 2: Hard Negative Samples (label = 1) ★ Core Novelty
    ความหมาย : ภาพที่ไม่มีวัตถุนั้น แต่มีวัตถุจาก Supercategory เดียวกัน
               จับคู่กับ Prompt ที่ตรงกับวัตถุที่ไม่มีอยู่ในภาพ
    ตัวอย่าง : ภาพที่มีสุนัขพันธุ์ Husky (แต่ไม่มีหมาป่า)
               + prompt "a photo of wolf"
               ทั้งสุนัขและหมาป่าอยู่ใน Supercategory "animal" เหมือนกัน
    วิธีสร้าง :
## 1. สำหรับแต่ละ Category X ให้หา Category อื่นที่อยู่ใน
         Supercategory เดียวกัน (Sibling Categories)
## 2. หาภาพที่มี Sibling Category แต่ไม่มี Category X
## 3. จับคู่ภาพนั้นกับ prompt "a photo of X"
    จำนวน    : 2,000 ตัวอย่าง
    ความสำคัญ : Hard Negatives เลียนแบบสถานการณ์จริงที่โมเดลมักจะ "หลอน"
                มากที่สุด เพราะวัตถุในภาพมีลักษณะคล้ายกับสิ่งที่ Prompt ถามหา
  ประเภทที่ 3: Easy Negative Samples (label = 1)
    ความหมาย : ภาพที่ไม่มีวัตถุนั้น และวัตถุในภาพมาจาก
               Supercategory ที่แตกต่างกันอย่างสิ้นเชิง
    ตัวอย่าง : ภาพที่มีเครื่องบิน + prompt "a photo of dog"
               เครื่องบินอยู่ใน "vehicle" ส่วนสุนัขอยู่ใน "animal"
    วิธีสร้าง : สุ่มเลือก Supercategory ที่แตกต่างจากวัตถุเป้าหมาย
               แล้วหาภาพจาก Supercategory นั้น
    จำนวน    : 1,000 ตัวอย่าง
3.3 โครงสร้าง Supercategory ของ COCO
  COCO จัด 80 Categories ออกเป็น 12 Supercategories เช่น:
    animal     : dog, cat, horse, bear, zebra, giraffe
    vehicle    : car, truck, bus, airplane, motorcycle
    furniture  : chair, couch, bed, dining table
    food       : banana, apple, sandwich, pizza
    outdoor    : traffic light, fire hydrant, stop sign
    (รวม 12 supercategories)
  การใช้โครงสร้าง Supercategory ทำให้เราสามารถสร้าง Hard Negatives
  ได้อย่างเป็นระบบ โดยไม่ต้องใช้ข้อมูลภายนอกหรือตัดสินใจด้วยมือ
---
## 4. Data Schema (โครงสร้างไฟล์ CSV)
---
ไฟล์ผลลัพธ์ hard_negative_dataset.csv มีโครงสร้างดังนี้:
  คอลัมน์         ประเภท      คำอธิบาย
  --------------- ----------- -------------------------------------------
  image_id        int         รหัสภาพจาก COCO (เช่น 139)
  image_path      str         ตำแหน่งไฟล์ภาพ (เช่น val2017/000000000139.jpg)
  prompt          str         คำสั่ง Prompt (เช่น "a photo of dog")
  true_label      int (0/1)   0 = ภาพมีวัตถุจริง, 1 = ภาพไม่มี (Hallucination)
  supercategory   str         กลุ่มใหญ่ของ COCO (เช่น animal, vehicle)
  negative_type   str         ประเภท: positive, hard, หรือ easy
---
## 5. การจัดการ Path ระหว่าง Local และ Colab (config.py)
---
ไฟล์ config.py ออกแบบมาเพื่อให้โค้ดทุกไฟล์ทำงานได้ทั้งบนเครื่อง Local
และ Google Colab โดยไม่ต้องแก้ Path ด้วยมือ:
  try:
      import google.colab
      IN_COLAB = True    # Path: /content/data
  except ImportError:
      IN_COLAB = False   # Path: D:/Advance Research/data
ทุกไฟล์ .py ใช้คำสั่ง import config แล้วเรียก config.get_base_dir()
เพื่อดึง Path ที่ถูกต้องตาม Environment อัตโนมัติ
---
## 6. Mockup CSV สำหรับทดสอบ
---
ก่อนที่จะรันชุดข้อมูลเต็มรูปแบบ (ต้องใช้ COCO จริง) ระบบสร้าง Mockup CSV
ข้อมูลจำลอง 10 แถว เพื่อให้สมาชิกคนอื่นทดสอบโค้ดได้ทันที
  คำสั่งรัน:
    python data/build_dataset.py --mode mockup   สร้าง Mockup 10 แถว
    python data/build_dataset.py --mode full     สร้างชุดข้อมูลจริง ~5,000 แถว
---
## 7. สิ่งที่ส่งมอบ (Deliverables)
---
  ไฟล์                         สถานะ         คำอธิบาย
  ---------------------------- ------------- ------------------------------------
  data/download_coco.py        เสร็จสมบูรณ์  สคริปต์ดาวน์โหลด COCO val2017
  data/build_dataset.py        เสร็จสมบูรณ์  สคริปต์สร้าง Dataset + Hard Negatives
  data/mockup_dataset.csv      เสร็จสมบูรณ์  ข้อมูลจำลอง 10 แถว (GitHub)
  config.py                    เสร็จสมบูรณ์  จัดการ Path อัตโนมัติ Local/Colab
  .gitignore                   เสร็จสมบูรณ์  ป้องกันอัปโหลดไฟล์ภาพ
  requirements.txt             เสร็จสมบูรณ์  รายการ Dependencies ทั้งหมด
---
## 8. Data Cleaning
---
COCO val2017 เป็นชุดข้อมูล Benchmark คุณภาพสูง Data Cleaning จึงมีน้อยมาก:
  สิ่งที่ทำ:
    - กรองภาพที่เสียหายหรืออ่านไม่ได้ (ถ้ามี)
    - ตรวจสอบว่า image_path ทุกแถวชี้ไปที่ไฟล์ที่มีอยู่จริง
    - สมดุลการกระจายตัว: positive vs hard vs easy
    - ตรวจสอบ Supercategory Mapping ตรงกับ COCO Format
  สิ่งที่ไม่ต้องทำ:
    - Annotation Cleaning (COCO เช็คให้แล้ว)
    - Image Normalization (HuggingFace Processor จัดการให้)
    - Missing Value Handling (ข้อมูล COCO ไม่มีค่า Missing)
---
## 9. Reproducibility
---
การสร้างข้อมูลทั้งหมด Reproducible ผ่าน Fixed Random Seed (--seed 42)
รันสคริปต์ด้วย Seed เดิม จะได้ CSV ที่เหมือนกันทุกประการ
  python data/build_dataset.py --mode full --seed 42

---

# ENGLISH VERSION
---
## 1. Role and Responsibilities
---
Person 1 serves as the Data Engineer, responsible for all data acquisition,
preprocessing, and dataset construction for the Hallucination Detection project.
The primary contribution is the design and implementation of a
supercategory-based hard negative sampling strategy using the COCO val2017
dataset, which constitutes the methodological novelty at the dataset level.
Files owned:
  - data/build_dataset.py     Core dataset construction script
  - data/download_coco.py     COCO download automation script
  - data/mockup_dataset.csv   10-row synthetic test dataset
  - config.py                 Dynamic path management for Local/Colab
  - requirements.txt          Project dependency manifest
  - .gitignore                Large file upload prevention
---
## 2. Data Source
---
The project uses COCO val2017 (Common Objects in Context) as the primary
data source.
  Dataset Name     : COCO val2017
  Total Images     : 5,000
  Annotation Type  : Instance Segmentation (Bounding Box + Category Label)
  Categories       : 80 object categories (e.g., dog, cat, car, airplane, chair)
  Supercategories  : 12 high-level groups (e.g., animal, vehicle, furniture, food)
  Annotation File  : instances_val2017.json (JSON format)
  Source           : https://cocodataset.org
Rationale for choosing COCO val2017:
  - Internationally recognised benchmark for Object Detection tasks
  - Built-in supercategory hierarchy enables hard negative generation
    without external data
  - High-quality, pre-verified annotations eliminate the need for
    extensive data cleaning
  - 5,000 images is a manageable yet statistically meaningful dataset
    size for the project timeline
---
## 3. Data Preparation Pipeline
---
3.1 Data Download (download_coco.py)
  The script downloads and extracts two file sets:
## 1. val2017.zip (~1 GB)                    5,000 validation images
## 2. annotations_trainval2017.zip (~241 MB)  Annotation files
  Key features:
    - Cross-environment compatibility (Local and Google Colab) via config.py
    - Automatic skip if files already exist (idempotent execution)
    - Automatic extraction and zip cleanup to save storage
    - Post-download verification step to confirm file counts
  Usage:
    python data/download_coco.py                          (Default path)
    python data/download_coco.py --data_dir /content/data  (Override for Colab)
3.2 Dataset Construction (build_dataset.py)
  This is the core deliverable of Person 1. The script reads
  instances_val2017.json via the pycocotools library and generates
  three types of samples:
  Type 1: Positive Samples (label = 0)
    Definition : Image genuinely contains object X, paired with
                 the correct prompt "a photo of X"
    Example    : Image containing a dog + prompt "a photo of dog"
    Method     : Random sampling from COCO images where the target
                 category is present
    Target     : 2,000 samples
  Type 2: Hard Negative Samples (label = 1) ★ Core Novelty
    Definition : Image contains object Y from the same supercategory
                 as X, but does NOT contain X. Paired with prompt
                 "a photo of X"
    Example    : Image of a Siberian Husky (no wolf present) +
                 prompt "a photo of wolf" — both dog and wolf belong
                 to the "animal" supercategory
    Method     :
## 1. For each category X, identify sibling categories within
         the same supercategory
## 2. Find images containing a sibling category but NOT category X
## 3. Pair the image with prompt "a photo of X"
    Target     : 2,000 samples
    Significance: Hard negatives simulate realistic hallucination
                  scenarios where visual similarity between the actual
                  object and the prompted object is high
  Type 3: Easy Negative Samples (label = 1)
    Definition : Image contains object Z from a completely different
                 supercategory, paired with prompt "a photo of X"
    Example    : Image of an airplane + prompt "a photo of dog" —
                 airplane is in "vehicle", dog is in "animal"
    Method     : Randomly select a different supercategory from the
                 target and sample images from it
    Target     : 1,000 samples
3.3 COCO Supercategory Hierarchy
  COCO organises its 80 object categories into 12 supercategories:
    animal     : dog, cat, horse, bear, zebra, giraffe
    vehicle    : car, truck, bus, airplane, motorcycle
    furniture  : chair, couch, bed, dining table
    food       : banana, apple, sandwich, pizza
    outdoor    : traffic light, fire hydrant, stop sign
    (12 supercategories total)
  Hard negatives are constructed by selecting prompts from within
  the same supercategory as the actual image content, ensuring high
  visual similarity between the true and prompted objects.
---
## 4. Data Schema
---
The output file hard_negative_dataset.csv follows the agreed schema:
  Column          Type        Description
  --------------- ----------- -------------------------------------------
  image_id        int         COCO image ID (e.g., 139)
  image_path      str         Relative path to image file
  prompt          str         Text prompt (e.g., "a photo of dog")
  true_label      int (0/1)   0 = genuine, 1 = hallucination expected
  supercategory   str         COCO supercategory (e.g., animal, vehicle)
  negative_type   str         Sample type: positive, hard, or easy
---
## 5. Cross-Environment Path Management (config.py)
---
The config.py module provides automatic environment detection and path
resolution, allowing all project scripts to run seamlessly on both
Local (Windows/macOS) and Google Colab without manual path modifications:
  Environment     Detected via                   Base Path
  --------------- ------------------------------ -------------------------
  Google Colab    import google.colab succeeds    /content/data
  Local (VS Code) import google.colab fails       <project_root>/data
All scripts accept an optional --data_dir CLI argument to override
the default path when needed.
---
## 6. Mockup CSV for Team Collaboration
---
Before running the full dataset builder (which requires actual COCO files),
the system can generate a Mockup CSV with 10 synthetic rows. This allows
other team members (Person 2, 3, 4) to immediately begin developing and
testing their respective modules without waiting for the full data download.
  Usage:
    python data/build_dataset.py --mode mockup   Generate 10-row test CSV
    python data/build_dataset.py --mode full     Generate full dataset
---
## 7. Deliverables Summary
---
  File                         Status       Description
  ---------------------------- ------------ ------------------------------------
  data/download_coco.py        Complete     COCO val2017 download script
  data/build_dataset.py        Complete     Full dataset builder + hard negatives
  data/mockup_dataset.csv      Complete     10-row mockup CSV (pushed to GitHub)
  config.py                    Complete     Dynamic path management
  .gitignore                   Complete     Prevents uploading large files
  requirements.txt             Complete     Project dependency list
---
## 8. Data Cleaning Assessment
---
COCO val2017 is a high-quality, well-maintained benchmark dataset.
Data cleaning workload is minimal:
  Required (light):
    - Filter corrupted or unreadable image files (if any)
    - Verify all image_path entries point to existing files
    - Balance class distribution: positive vs hard vs easy
    - Validate supercategory mapping matches COCO format
  Not required:
    - Annotation cleaning (COCO annotations are pre-verified)
    - Image normalisation (handled by HuggingFace processors)
    - Missing value handling (structured dataset with no missing entries)
---
## 9. Reproducibility
---
All data generation is fully reproducible via a fixed random seed
(--seed 42 by default). Running the same script with the same seed
will produce an identical CSV file.
  python data/build_dataset.py --mode full --seed 42