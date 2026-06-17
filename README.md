# AutoSmokeGuard — Backend

Django REST API server for the AutoSmokeGuard vehicle smoke detection and emission-report system.

## Tech Stack

- **Django** + **Django REST Framework** – RESTful API backend
- **PostgreSQL** – Relational database (planned for later sprints)
- **PyTorch / YOLO / OpenCV** – Machine learning and computer vision (planned for later sprints)

## Getting Started

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the development server:
   ```bash
   python manage.py runserver
   ```

## Sprint Scope

- **Sprint 1** – Django project skeleton with `/health` endpoint only
- **Sprint 2-4** – Authentication APIs, database models, and foundational structures
- **Sprint 5+** – Machine learning integration, image upload processing, emission analysis, reporting
