# models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    dob = db.Column(db.Date, nullable=True) 

    transactions = db.relationship('Transaction', backref='user', lazy=True)
    schemes = db.relationship('FixedScheme', backref='user', lazy=True)
    salary_details = db.relationship('Salary', backref='user', uselist=False)
    investments = db.relationship('Investment', backref='user', lazy=True)
    sold_investments = db.relationship('SoldInvestment', backref='user', lazy=True)
    # --- NEW RELATIONSHIP ---
    loans = db.relationship('Loan', backref='user', lazy=True)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(200), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(10), nullable=False)
    category = db.Column(db.String(50), nullable=False, default='Uncategorized')
    date = db.Column(db.Date, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class FixedScheme(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    scheme_name = db.Column(db.String(200), nullable=False)
    principal_amount = db.Column(db.Float, nullable=False)
    interest_rate = db.Column(db.Float, nullable=False)
    tenure_months = db.Column(db.Integer, nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    penalty_rate = db.Column(db.Float, nullable=False, default=1.0)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Salary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    monthly_gross = db.Column(db.Float, nullable=False, default=0)
    deductions_80c = db.Column(db.Float, nullable=False, default=0)
    hra_exemption = db.Column(db.Float, nullable=False, default=0)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)

class Investment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    asset_type = db.Column(db.String(20), nullable=False)
    ticker_symbol = db.Column(db.String(50), nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    purchase_price = db.Column(db.Float, nullable=False)
    purchase_currency = db.Column(db.String(10), nullable=False, default='INR')
    purchase_date = db.Column(db.Date, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class SoldInvestment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    asset_type = db.Column(db.String(20), nullable=False)
    ticker_symbol = db.Column(db.String(50), nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    purchase_price = db.Column(db.Float, nullable=False)
    purchase_date = db.Column(db.Date, nullable=False)
    sell_price = db.Column(db.Float, nullable=False)
    sell_date = db.Column(db.Date, nullable=False)
    capital_gain = db.Column(db.Float, nullable=False)
    gain_type = db.Column(db.String(10), nullable=False) # STCG or LTCG
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# --- NEW TABLE ---
class Loan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    loan_name = db.Column(db.String(100), nullable=False)
    principal = db.Column(db.Float, nullable=False)
    interest_rate = db.Column(db.Float, nullable=False)
    tenure_months = db.Column(db.Integer, nullable=False)
    emi_amount = db.Column(db.Float, nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
