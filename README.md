# 🚚 Delivery Delay Dashboard with Role-Based Access

A role-based Streamlit dashboard for delivery delay analytics. Provides tailored views for Executives (KPIs), Logistics Managers (root causes), and Customer Service (order lookup).

## 🚀 Live Demo
**Experience the dashboard here:**  
👉 [https://supply-chain-delivery-delay-dashboard.streamlit.app/](https://supply-chain-delivery-delay-dashboard.streamlit.app/)

## ✨ Features
- **🔐 Role-Based Access Control** (Executive, Logistics Manager, Customer Service)
- **🎛️ Interactive Filters** by date and other dimensions
- **📊 Performance Metrics** and delay analysis
- **📱 Responsive Design** built with Streamlit

## 🏗️ Role Access
| Role | Access Level | Key Features |
|------|-------------|--------------|
| **👔 Executive** | Strategic | High-level KPIs and trends |
| **📦 Logistics Manager** | Operational | Carrier performance and root cause analysis |
| **📞 Customer Service** | Tactical | Order status lookup and delay verification |

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/amosjacobaboy/delivery-delay-dashboard.git

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run streamlit_app.py
