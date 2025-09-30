from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

db = SQLAlchemy()

# --- CORE SHARED MODEL ---

class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(10), nullable=False)  # 'income' or 'expense'
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Explicitly define relationships from the "one" side (Category) to the "many" sides
    transactions = db.relationship('Transaction', back_populates='category', cascade="all, delete-orphan")
    business_transactions = db.relationship('BusinessTransaction', back_populates='category', cascade="all, delete-orphan")
    budgets = db.relationship('Budget', backref='category', lazy=True, cascade="all, delete-orphan")


# --- PERSONAL & BUSINESS USER MODEL ---

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    dob = db.Column(db.Date, nullable=True)
    role = db.Column(db.String(50), nullable=False, default='employee')
    
    # Personal finance relationships
    transactions = db.relationship('Transaction', backref='user', lazy=True, cascade="all, delete-orphan")
    schemes = db.relationship('FixedScheme', backref='user', lazy=True, cascade="all, delete-orphan")
    salary_details = db.relationship('Salary', backref='user', uselist=False, cascade="all, delete-orphan")
    investments = db.relationship('Investment', backref='user', lazy=True, cascade="all, delete-orphan")
    sold_investments = db.relationship('SoldInvestment', backref='user', lazy=True, cascade="all, delete-orphan")
    loans = db.relationship('Loan', backref='user', lazy=True, cascade="all, delete-orphan")
    categories = db.relationship('Category', backref='user', lazy=True, cascade="all, delete-orphan")
    budgets = db.relationship('Budget', backref='user', lazy=True, cascade="all, delete-orphan")

    # Business relationships
    business_transactions = db.relationship('BusinessTransaction', backref='user', lazy=True, cascade="all, delete-orphan")
    business_investments = db.relationship('BusinessInvestment', backref='user', lazy=True, cascade="all, delete-orphan")
    business_loans = db.relationship('BusinessLoan', backref='user', lazy=True, cascade="all, delete-orphan")
    business_clients = db.relationship('BusinessClient', backref='user', lazy=True, cascade="all, delete-orphan")
    business_metrics = db.relationship('BusinessMetrics', backref='user', uselist=False, cascade="all, delete-orphan")


# --- PERSONAL FINANCE MODELS ---

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(200), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(10), nullable=False)
    date = db.Column(db.Date, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('category.id'), nullable=False)
    
    # Explicitly link back to the 'transactions' property in the Category model
    category = db.relationship('Category', back_populates='transactions')

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

class Loan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    loan_name = db.Column(db.String(100), nullable=False)
    principal = db.Column(db.Float, nullable=False)
    interest_rate = db.Column(db.Float, nullable=False)
    tenure_months = db.Column(db.Integer, nullable=False)
    emi_amount = db.Column(db.Float, nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


# --- BUDGET MODEL ---

class Budget(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category_id = db.Column(db.Integer, db.ForeignKey('category.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    year = db.Column(db.Integer, nullable=False)

    __table_args__ = (db.UniqueConstraint('user_id', 'category_id', 'month', 'year', name='_user_category_month_year_uc'),)


# --- BUSINESS MODELS ---

class BusinessMetrics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True)
    monthly_revenue = db.Column(db.Float, default=0.0)
    monthly_expenses = db.Column(db.Float, default=0.0)
    net_profit = db.Column(db.Float, default=0.0)
    profit_margin = db.Column(db.Float, default=0.0)

class BusinessTransaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    description = db.Column(db.String(200), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(10), nullable=False)
    date = db.Column(db.Date, nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('category.id'), nullable=False)
    receipt_filename = db.Column(db.String(255), nullable=True)

    # Explicitly link back to the 'business_transactions' property in the Category model
    category = db.relationship('Category', back_populates='business_transactions')

class BusinessInvestment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    investment_name = db.Column(db.String(100))
    investment_type = db.Column(db.String(50))
    amount_invested = db.Column(db.Float)
    purchase_date = db.Column(db.Date)
    useful_life_years = db.Column(db.Integer, nullable=True)

    @property
    def current_value(self):
        if self.investment_type != 'Financial' and self.useful_life_years and self.useful_life_years > 0:
            years_owned = (date.today() - self.purchase_date).days / 365.25
            if years_owned >= self.useful_life_years: return 0.0
            depreciation_per_year = self.amount_invested / self.useful_life_years
            current_val = self.amount_invested - (depreciation_per_year * years_owned)
            return max(0, current_val)
        else:
            return self.amount_invested

    @property
    def roi(self):
        if self.amount_invested > 0:
            return ((self.current_value - self.amount_invested) / self.amount_invested) * 100
        return 0.0

class BusinessLoan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    loan_name = db.Column(db.String(100))
    principal_amount = db.Column(db.Float)
    interest_rate = db.Column(db.Float)
    tenure_months = db.Column(db.Integer)
    start_date = db.Column(db.Date)
    emi = db.Column(db.Float)

    @property
    def remaining_balance(self):
        r = (self.interest_rate / 12) / 100
        n = self.tenure_months
        p_made = (date.today().year - self.start_date.year) * 12 + date.today().month - self.start_date.month
        if p_made < 0: p_made = 0
        if p_made >= n: return 0.0
        balance = self.principal_amount * (((1 + r)**n) - ((1 + r)**p_made)) / (((1 + r)**n) - 1)
        return balance if balance > 0 else 0.0

class BusinessClient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    client_name = db.Column(db.String(100))
    project = db.Column(db.String(150))
    revenue_contribution = db.Column(db.Float)
    status = db.Column(db.String(50))