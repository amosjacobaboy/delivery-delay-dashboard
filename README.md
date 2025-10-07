📦 Delivery Delay Analytics Dashboard
A role-based Streamlit dashboard for analyzing delivery delay factors in supply chain operations.

🎯 Overview
This application provides tailored insights into delivery delay patterns for different business roles:

Executives: Strategic insights and high-level recommendations

Logistics Managers: Operational analysis and improvement plans

Customer Service: Customer impact and communication strategies

🚀 Features
🔐 Enhanced Authentication
Role-based access control

Secure password hashing

Three user roles with different permissions

📊 Role-Based Analytics
Executive View: Strategic business impact and ROI analysis

Logistics View: Operational factors and tactical action plans

Customer Service View: Customer impact and communication strategies

🔍 Research Question 1 Analysis
Comprehensive factor correlation analysis

Role-specific visualizations

Actionable recommendations for each user type

📈 Advanced Analytics
Delivery delay prediction factors

Geographic and temporal analysis

Performance metrics and KPIs

📁 Project Structure
text
delivery-delay-app/
├── delivery-delay-app3.py      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── merged-dataset.csv          # Input data file (required)
🛠️ Installation
Clone or download the project files

Install dependencies:

bash
pip install -r requirements.txt
Prepare your data:

Ensure you have a merged-dataset.csv file in the same directory

The dataset should contain delivery and order information

🎮 Usage
Run the application:

bash
streamlit run delivery-delay-app3.py
Login with one of the following credentials:

Role	Username	Password
Executive	executive	exec123
Logistics Manager	logistics	logi123
Customer Service	custsvc	cust123
Explore role-specific dashboards:

Each role sees tailored visualizations and insights

Navigate through different tabs for comprehensive analysis

📊 Data Requirements
The application expects a CSV file with the following columns (at minimum):

order_purchase_timestamp

order_approved_at

order_delivered_carrier_date

order_delivered_customer_date

order_estimated_delivery_date

order_id, customer_id, seller_id

price, seller_zip_code_prefix, customer_zip_code_prefix

🔧 Technical Features
Caching: Efficient data loading with @st.cache_data

Responsive Design: Wide layout optimized for analytics

Interactive Visualizations: Plotly charts with hover details

Real-time Metrics: Dynamic KPI calculations

Security: SHA-256 password hashing

👥 User Roles & Capabilities
Executive
Strategic business impact analysis

ROI calculations and investment recommendations

High-level performance metrics

Executive action plans

Logistics Manager
Operational factor analysis

Carrier performance insights

Tactical improvement plans

Process optimization recommendations

Customer Service
Customer impact assessment

Regional delay patterns

Communication strategy templates

Proactive support planning

📈 Analytical Capabilities
Delivery delay rate calculations

Factor correlation analysis

Geographic performance mapping

Temporal pattern recognition

Predictive factor identification

🎨 Customization
The application can be customized by:

Modifying the user credentials in the USERS dictionary

Adjusting the feature engineering logic in load_and_enhance_data()

Customizing the visualizations and metrics for each role

Adding new analysis modules for additional research questions

⚠️ Notes
Ensure your dataset follows the expected format for proper analysis

The application automatically handles date conversions and data cleaning

All passwords are hashed using SHA-256 for security

The dashboard is optimized for desktop use

📞 Support
For technical issues or questions about the analytics, refer to the code comments or contact the development team.

Group 3 PORA Supply Chain Solutions © 2025