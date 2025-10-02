---
title: AI-Powered Finance Tracker
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
app_port: 7860
---

# 🤖 Fin-AI: An Intelligent Personal & Business Finance Manager

**An advanced, dual-purpose financial management application built with Flask and Python, leveraging AI for intelligent data extraction and machine learning for proactive financial insights.**

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/your_username/your_project_repo)

This project demonstrates a comprehensive, full-stack web application designed to solve real-world financial tracking challenges through modern technology.


*(Replace this with a high-quality screenshot of your application's most impressive page)*

---

## 🎯 Project Motivation & Problem Statement

In a market saturated with generic expense trackers, many users still struggle with the tedious task of manual data entry and the lack of personalized, actionable insights. Personal finance is not one-size-fits-all, and the needs of an individual are vastly different from those of a freelancer or small business owner.

This project tackles these problems by:
1.  **Automating Data Entry:** Utilizing AI to scan and parse receipts, drastically reducing the time and effort required to log transactions.
2.  **Providing Proactive Insights:** Employing machine learning to detect spending anomalies in real-time, offering a layer of financial security and awareness that most manual trackers lack.
3.  **Serving a Dual Role:** Offering a tailored experience for both personal and business finance within a single, cohesive platform, recognizing that many individuals manage both.

The goal of Fin-AI is to create a smarter, more intuitive financial companion that adapts to the user's habits and provides genuine value beyond simple record-keeping.

---

## 🏗️ System Architecture

The application is built on a classic three-tier architecture, designed for scalability and maintainability. External services are integrated to handle specialized tasks like media storage and AI processing.

[Diagram of the application architecture]
*(A diagram showing: User -> Browser -> Flask Web Server -> (SQL Database | Cloudinary API | Gemini AI API))*

1.  **Client-Side (Frontend):** A responsive user interface built with HTML, Bootstrap 5, and vanilla JavaScript that runs in the user's browser. It handles user interaction and makes asynchronous calls (AJAX/Fetch) to the backend for dynamic features like receipt scanning.
2.  **Server-Side (Backend):** A robust Flask application serves as the core of the system. It handles business logic, user authentication, database interactions, and communication with external APIs.
3.  **Data Layer:**
    * A relational database (e.g., MySQL) managed by SQLAlchemy ORM stores all core user data, including transactions, categories, budgets, and loans.
    * **Cloudinary** is used as an object storage service to persistently and securely store all user-uploaded media (receipts), keeping them separate from the application's ephemeral filesystem.

---

## ✨ Features in Detail

 🤖 AI-Powered Receipt Scanner
This feature revolutionizes transaction entry.
* **Workflow:** A user uploads a receipt image via the web interface. A client-side JavaScript function sends this image to a dedicated Flask endpoint.
* **Technology:** The backend uploads the image to **Cloudinary** for persistent storage, receiving a secure URL in return. This image data is then passed to an AI model (e.g., Google Gemini or Donut) which extracts key information like the vendor, total amount, and date.
* **User Experience:** The extracted data, along with the Cloudinary URL, is sent back to the browser as a JSON response. The JavaScript then automatically populates the fields of the "Add Transaction" form, allowing the user to simply verify the data and click save.

 📈 ML-Powered Anomaly Detection
To provide proactive financial security, the system includes a custom anomaly detection model.
* **Model:** A `IsolationForest` model from the Scikit-learn library is trained on a per-user basis using their historical transaction data.
* **Implementation:** After a new business transaction is saved, it is passed to a `check_transaction_anomaly` function. The function uses the pre-trained model to predict whether the new transaction is an "inlier" (normal) or an "outlier" (unusual).
* **User Feedback:** If an outlier is detected, a prominent `flash` message is displayed on the dashboard, warning the user of the unusually high transaction and prompting them to review it.
* **Auto-Retraining:** The model is automatically retrained with fresh data in the background after every 20 new transactions, ensuring it stays up-to-date with the user's evolving spending patterns.

 💰 Smart Budgeting System
This module empowers users to take control of their spending.
* **Functionality:** Users can navigate to a dedicated Budget Planner page where all their 'expense' type categories are listed. They can set a monthly spending limit for any or all of these categories.
* **Visualization:** The system provides real-time feedback with color-coded progress bars (green -> yellow -> red) that show how much of the budget has been consumed. A summary card on the main dashboard gives an at-a-glance overview of the overall budget status.

---

## 🛠️ Tech Stack

### **Backend**
* **Framework:** Flask
* **Database:** SQLAlchemy ORM, Flask-SQLAlchemy
* **Migrations:** Flask-Migrate
* **Authentication:** Flask-Login
* **Security:** Flask-WTF (CSRF Protection)
* **Language:** Python 3.10+

### **Frontend**
* **Templating:** Jinja2
* **Styling:** Bootstrap 5
* **Scripting:** Vanilla JavaScript (Fetch API for AJAX)

### **AI & Machine Learning**
* **ML Library:** Scikit-learn
* **Cloud Media Storage:** Cloudinary API
* **AI Data Extraction:** Custom AI Model Integration (Gemini, Donut, etc.)

### **Development & Deployment**
* **Environment:** Python `venv`
* **Configuration:** Environment Variables (`.env` files with `python-dotenv`)

---

## 🚀 Getting Started

To get a local copy up and running, follow these steps.

### **Prerequisites**
* Python 3.8+
* A running SQL database (e.g., MySQL, PostgreSQL)

### **Installation & Setup**
1.  Clone the repository:
    ```bash
    git clone [https://github.com/your_username/your_project_repo.git](https://github.com/your_username/your_project_repo.git)
    cd your_project_repo
    ```
2.  Create and activate a virtual environment:
    ```bash
    # Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  Install the required packages from `requirements.txt`:
    *(If you don't have this file, create it by running `pip freeze > requirements.txt`)*
    ```bash
    pip install -r requirements.txt
    ```
4.  Configure Environment Variables by creating a `.env` file in the project root:
    ```env
    # .env file

    SECRET_KEY='a-very-long-and-random-secret-key'
    SQLALCHEMY_DATABASE_URI='mysql+pymysql://user:password@hostname/db_name'
    CLOUDINARY_CLOUD_NAME='your_cloud_name'
    CLOUDINARY_API_KEY='your_api_key'
    CLOUDINARY_API_SECRET='your_api_secret'
    ```
5.  Initialize and run database migrations:
    ```bash
    flask db init  # (Only if you haven't already)
    flask db migrate -m "Initial database migration."
    flask db upgrade
    ```
6.  Run the application:
    ```bash
    flask run
    ```

---

## 🗺️ API Endpoints

The application exposes several key endpoints for both web navigation and AJAX calls.

| Method | Endpoint                      | Description                                                  |
| :----- | :---------------------------- | :----------------------------------------------------------- |
| GET, POST | `/categories`                | Manages (view, add, delete) user-defined categories.        |
| GET, POST | `/budget`                    | Manages (view, set) the monthly budget for expense categories. |
| GET, POST | `/business/transactions`     | View transaction history and add a new business transaction. |
| POST    | `/process-receipt-accurate`  | AJAX endpoint for the AI to scan a receipt and return JSON data. |
| GET     | `/dashboard`                 | Displays the personal finance dashboard.                     |
| GET     | `/business_dashboard`        | Displays the business finance dashboard.                     |

---

## 🛣️ Project Roadmap

This project has a solid foundation, but there are many exciting features that could be added in the future:

* [ ] **Advanced Reporting:** Create a dedicated "Reports" page with interactive charts (using Chart.js) for visualizing spending trends, category breakdowns, and net worth over time.
* [ ] **Recurring Transactions:** Allow users to set up automatic recurring transactions for bills and subscriptions.
* [ ] **Debt Paydown Calculator:** A tool to help users visualize how extra payments can impact their loan repayment timelines.
* [ ] **Investment Portfolio Integration:** Connect to a financial API (e.g., Finnhub) to track the real-time value of stock and crypto investments.
* [ ] **Email Notifications:** Send users weekly summaries or alerts when they are approaching a budget limit.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📧 Contact

Your Name - [@YourTwitter](https://twitter.com/YourTwitter) - your.email@example.com

Project Link: [https://github.com/your_username/your_project_repo](https://github.com/your_username/your_project_repo)
