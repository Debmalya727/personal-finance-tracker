# app.py

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Transaction, FixedScheme, Salary, Investment, SoldInvestment, Loan
from datetime import datetime, date
import json
from dateutil.relativedelta import relativedelta
import yfinance as yf
from pycoingecko import CoinGeckoAPI
import os
from dotenv import load_dotenv
from flask_wtf.csrf import CSRFProtect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
from flask_migrate import Migrate

load_dotenv()
app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

csrf = CSRFProtect(app)
db.init_app(app)
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- Tax Helper Functions ---
def calculate_new_regime_tax(gross_income):
    standard_deduction = 50000
    taxable_income = gross_income - standard_deduction
    if taxable_income < 0: taxable_income = 0
    tax = 0
    if taxable_income <= 300000: tax = 0
    elif taxable_income <= 600000: tax = (taxable_income - 300000) * 0.05
    elif taxable_income <= 900000: tax = 15000 + (taxable_income - 600000) * 0.10
    elif taxable_income <= 1200000: tax = 45000 + (taxable_income - 900000) * 0.15
    elif taxable_income <= 1500000: tax = 90000 + (taxable_income - 1200000) * 0.20
    else: tax = 150000 + (taxable_income - 1500000) * 0.30
    cess = tax * 0.04
    total_tax = tax + cess
    return {'regime': 'New', 'gross_income': gross_income, 'taxable_income': taxable_income, 'total_deductions': 0, 'standard_deduction': standard_deduction, 'income_tax': tax, 'cess': cess, 'total_tax': total_tax}

def calculate_old_regime_tax(gross_income, total_deductions, age):
    standard_deduction = 50000
    taxable_income = gross_income - total_deductions - standard_deduction
    if taxable_income < 0: taxable_income = 0
    tax = 0
    if age < 60:
        if taxable_income <= 250000: tax = 0
        elif taxable_income <= 500000: tax = (taxable_income - 250000) * 0.05
        elif taxable_income <= 1000000: tax = 12500 + (taxable_income - 500000) * 0.20
        else: tax = 112500 + (taxable_income - 1000000) * 0.30
    elif age < 80:
        if taxable_income <= 300000: tax = 0
        elif taxable_income <= 500000: tax = (taxable_income - 300000) * 0.05
        elif taxable_income <= 1000000: tax = 10000 + (taxable_income - 500000) * 0.20
        else: tax = 110000 + (taxable_income - 1000000) * 0.30
    else:
        if taxable_income <= 500000: tax = 0
        elif taxable_income <= 1000000: tax = (taxable_income - 500000) * 0.20
        else: tax = 100000 + (taxable_income - 1000000) * 0.30
    if taxable_income <= 500000: tax = 0
    cess = tax * 0.04
    total_tax = tax + cess
    return {'regime': 'Old', 'gross_income': gross_income, 'taxable_income': taxable_income, 'total_deductions': total_deductions, 'standard_deduction': standard_deduction, 'income_tax': tax, 'cess': cess, 'total_tax': total_tax}

# --- Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        if 'register' in request.form:
            username = request.form.get('username'); password = request.form.get('password'); dob_str = request.form.get('dob')
            if User.query.filter_by(username=username).first(): flash('Username already exists.', 'error'); return redirect(url_for('login'))
            dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(username=username, password=hashed_password, dob=dob)
            db.session.add(new_user); db.session.commit()
            flash('Registration successful! Please log in.', 'success'); return redirect(url_for('login'))
        elif 'login' in request.form:
            username = request.form.get('username'); password = request.form.get('password')
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password, password): login_user(user); return redirect(url_for('dashboard'))
            else: flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user(); return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    today = date.today()
    start_of_month = today.replace(day=1)
    
    salary_details = Salary.query.filter_by(user_id=current_user.id).first()
    if salary_details and salary_details.monthly_gross > 0:
        salary_credited_this_month = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            Transaction.description == "Monthly Salary",
            Transaction.date >= start_of_month
        ).first()
        if not salary_credited_this_month:
            salary_transaction = Transaction(
                description="Monthly Salary",
                amount=salary_details.monthly_gross,
                type="income",
                category="Salary",
                date=start_of_month,
                user_id=current_user.id
            )
            db.session.add(salary_transaction)
            db.session.commit()
            flash(f"Auto-credited salary of ₹{salary_details.monthly_gross} for this month.", "info")

    user_loans = Loan.query.filter_by(user_id=current_user.id).all()
    for loan in user_loans:
        emi_debited = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            Transaction.description == f"EMI for {loan.loan_name}",
            Transaction.date >= start_of_month
        ).first()
        
        loan_end_date = loan.start_date + relativedelta(months=+loan.tenure_months)
        if not emi_debited and today <= loan_end_date:
            db.session.add(Transaction(description=f"EMI for {loan.loan_name}", amount=loan.emi_amount, type="expense", category="EMI", date=start_of_month, user_id=current_user.id))
            db.session.commit()
            flash(f"Auto-debited EMI of ₹{loan.emi_amount} for {loan.loan_name}.", "info")

    monthly_transactions = Transaction.query.filter(
        Transaction.user_id == current_user.id,
        Transaction.date >= start_of_month
    ).all()
    monthly_income = sum(t.amount for t in monthly_transactions if t.type == 'income')
    monthly_expense = sum(t.amount for t in monthly_transactions if t.type == 'expense')

    all_transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    balance = sum(t.amount for t in all_transactions if t.type == 'income') - sum(t.amount for t in all_transactions if t.type == 'expense')
    
    recent_transactions = Transaction.query.filter_by(user_id=current_user.id).order_by(Transaction.date.desc()).limit(5).all()

    chart_data = {'labels': ['Monthly Income', 'Monthly Expense'], 'data': [monthly_income, monthly_expense]}
    
    return render_template('dashboard.html', 
                           user=current_user, 
                           balance=balance, 
                           monthly_income=monthly_income, 
                           monthly_expense=monthly_expense, 
                           recent_transactions=recent_transactions,
                           chart_data=json.dumps(chart_data))

@app.route('/add_transaction', methods=['POST'])
@login_required
def add_transaction():
    description = request.form.get('description'); amount = float(request.form.get('amount')); ttype = request.form.get('type'); category = request.form.get('category')
    new_transaction = Transaction(description=description, amount=amount, type=ttype, category=category, date=datetime.utcnow().date(), user_id=current_user.id)
    db.session.add(new_transaction); db.session.commit()
    flash('Transaction added successfully!', 'success'); return redirect(url_for('dashboard'))

@app.route('/transactions')
@login_required
def view_transactions():
    transactions = Transaction.query.filter_by(user_id=current_user.id).order_by(Transaction.date.desc()).all()
    return render_template('transactions.html', transactions=transactions)

@app.route('/delete_transaction/<int:transaction_id>', methods=['POST'])
@login_required
def delete_transaction(transaction_id):
    transaction = db.session.get(Transaction, transaction_id)
    if not transaction or transaction.user_id != current_user.id: flash('Not authorized.', 'error'); return redirect(url_for('view_transactions'))
    db.session.delete(transaction); db.session.commit()
    flash('Transaction deleted.', 'success'); return redirect(url_for('view_transactions'))

@app.route('/schemes')
@login_required
def schemes():
    user_schemes = FixedScheme.query.filter_by(user_id=current_user.id).all()
    schemes_with_details = []
    for scheme in user_schemes:
        years_elapsed = (date.today() - scheme.start_date).days / 365.25
        if years_elapsed < 0: years_elapsed = 0
        tenure_years = scheme.tenure_months / 12
        maturity_amount = scheme.principal_amount * ((1 + (scheme.interest_rate / 100)) ** tenure_years)
        current_value = scheme.principal_amount * ((1 + (scheme.interest_rate / 100)) ** years_elapsed)
        penalized_rate = scheme.interest_rate - scheme.penalty_rate
        if penalized_rate < 0: penalized_rate = 0
        early_withdrawal_value = scheme.principal_amount * ((1 + (penalized_rate / 100)) ** years_elapsed)
        maturity_date = scheme.start_date + relativedelta(months=+scheme.tenure_months)
        schemes_with_details.append({'scheme': scheme, 'maturity_date': maturity_date, 'maturity_amount': maturity_amount, 'current_value': current_value, 'early_withdrawal_value': early_withdrawal_value})
    return render_template('schemes.html', schemes_data=schemes_with_details)

@app.route('/add_scheme', methods=['POST'])
@login_required
def add_scheme():
    scheme_name = request.form.get('scheme_name'); principal = float(request.form.get('principal_amount')); rate = float(request.form.get('interest_rate')); tenure = int(request.form.get('tenure_months')); start_date_str = request.form.get('start_date'); penalty = float(request.form.get('penalty_rate'))
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    new_scheme = FixedScheme(scheme_name=scheme_name, principal_amount=principal, interest_rate=rate, tenure_months=tenure, start_date=start_date, penalty_rate=penalty, user_id=current_user.id)
    db.session.add(new_scheme); db.session.commit()
    flash(f'Scheme "{scheme_name}" added successfully!', 'success'); return redirect(url_for('schemes'))

@app.route('/delete_scheme/<int:scheme_id>', methods=['POST'])
@login_required
def delete_scheme(scheme_id):
    scheme = db.session.get(FixedScheme, scheme_id)
    if not scheme or scheme.user_id != current_user.id: flash('Not authorized.', 'error'); return redirect(url_for('schemes'))
    db.session.delete(scheme); db.session.commit()
    flash('Scheme deleted.', 'success'); return redirect(url_for('schemes'))

@app.route('/salary_manager', methods=['GET', 'POST'])
@login_required
def salary_manager():
    salary_details = Salary.query.filter_by(user_id=current_user.id).first()
    if request.method == 'POST':
        monthly_gross = float(request.form.get('monthly_gross'))
        deductions_80c = float(request.form.get('deductions_80c'))
        hra_exemption = float(request.form.get('hra_exemption'))
        if salary_details:
            salary_details.monthly_gross = monthly_gross; salary_details.deductions_80c = deductions_80c; salary_details.hra_exemption = hra_exemption
            flash('Salary details updated successfully!', 'success')
        else:
            salary_details = Salary(monthly_gross=monthly_gross, deductions_80c=deductions_80c, hra_exemption=hra_exemption, user_id=current_user.id)
            db.session.add(salary_details)
            flash('Salary details saved successfully!', 'success')
        db.session.commit()
        return redirect(url_for('salary_manager'))
    return render_template('salary_manager.html', salary=salary_details)

@app.route('/investments')
@login_required
def investments():
    sold_investments = SoldInvestment.query.filter_by(user_id=current_user.id).order_by(SoldInvestment.sell_date.desc()).all()
    return render_template('investments.html', sales=sold_investments)

@app.route('/add_investment', methods=['POST'])
@login_required
def add_investment():
    asset_type = request.form.get('asset_type')
    ticker = request.form.get('ticker_symbol').lower() if asset_type == 'Crypto' else request.form.get('ticker_symbol').upper()
    quantity = float(request.form.get('quantity'))
    price = float(request.form.get('purchase_price'))
    currency = request.form.get('purchase_currency')
    purchase_date_str = request.form.get('purchase_date')
    purchase_date = datetime.strptime(purchase_date_str, '%Y-%m-%d').date()
    new_investment = Investment(asset_type=asset_type, ticker_symbol=ticker, quantity=quantity, purchase_price=price, purchase_currency=currency, purchase_date=purchase_date, user_id=current_user.id)
    db.session.add(new_investment); db.session.commit()
    flash(f'{asset_type} "{ticker}" added to your portfolio!', 'success'); return redirect(url_for('investments'))

@app.route('/delete_investment/<int:investment_id>', methods=['POST'])
@login_required
def delete_investment(investment_id):
    investment = db.session.get(Investment, investment_id)
    if not investment or investment.user_id != current_user.id:
        flash('Not authorized to delete this investment.', 'error'); return redirect(url_for('investments'))
    db.session.delete(investment); db.session.commit()
    flash('Investment removed from portfolio.', 'success'); return redirect(url_for('investments'))

@app.route('/refresh_prices')
@login_required
def refresh_prices():
    user_investments = Investment.query.filter_by(user_id=current_user.id).all()
    refreshed_data = []
    cg = CoinGeckoAPI()
    try:
        rates = cg.get_price(ids='tether', vs_currencies='inr')
        usd_to_inr_rate = rates['tether']['inr']
    except Exception as e:
        print(f"Could not fetch exchange rate, falling back. Error: {e}")
        usd_to_inr_rate = 83.5
    for investment in user_investments:
        current_price_display = 0; total_value_inr = 0; profit_loss_display = 0
        try:
            if investment.asset_type == 'Stock':
                stock = yf.Ticker(investment.ticker_symbol)
                todays_data = stock.history(period='1d')
                if not todays_data.empty:
                    current_price_display = todays_data['Close'][0]
                    total_value_inr = investment.quantity * current_price_display
                    investment_cost_inr = investment.purchase_price * investment.quantity
                    profit_loss_display = total_value_inr - investment_cost_inr
            elif investment.asset_type == 'Crypto':
                price_data = cg.get_price(ids=investment.ticker_symbol, vs_currencies='usd')
                if price_data and price_data.get(investment.ticker_symbol):
                    current_price_display = price_data[investment.ticker_symbol].get('usd', 0)
                    total_value_inr = (investment.quantity * current_price_display) * usd_to_inr_rate
                    investment_cost_usd = investment.purchase_price * investment.quantity
                    if investment.purchase_currency == 'INR':
                        investment_cost_usd /= usd_to_inr_rate
                    profit_loss_display = (investment.quantity * current_price_display) - investment_cost_usd
        except Exception as e:
            print(f"Could not fetch price for {investment.ticker_symbol}: {e}")
        refreshed_data.append({'investment': {'id': investment.id, 'ticker_symbol': investment.ticker_symbol.upper(), 'asset_type': investment.asset_type, 'quantity': investment.quantity, 'purchase_price': investment.purchase_price, 'purchase_currency': investment.purchase_currency}, 'current_price_display': current_price_display, 'total_value_inr': total_value_inr, 'profit_loss_display': profit_loss_display})
    return jsonify({'data': refreshed_data, 'exchange_rate': usd_to_inr_rate})

@app.route('/net_worth')
@login_required
def net_worth():
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    cash_balance = sum(t.amount for t in transactions if t.type == 'income') - sum(t.amount for t in transactions if t.type == 'expense')
    user_schemes = FixedScheme.query.filter_by(user_id=current_user.id).all()
    total_schemes_value = 0
    for scheme in user_schemes:
        years_elapsed = (date.today() - scheme.start_date).days / 365.25
        if years_elapsed > 0:
            total_schemes_value += scheme.principal_amount * ((1 + (scheme.interest_rate / 100)) ** years_elapsed)
    user_investments = Investment.query.filter_by(user_id=current_user.id).all()
    total_investments_value = 0
    cg = CoinGeckoAPI()
    try:
        rates = cg.get_price(ids='tether', vs_currencies='inr')
        usd_to_inr_rate = rates['tether']['inr']
    except Exception:
        usd_to_inr_rate = 83.5
    for investment in user_investments:
        try:
            if investment.asset_type == 'Stock':
                stock = yf.Ticker(investment.ticker_symbol)
                todays_data = stock.history(period='1d')
                if not todays_data.empty:
                    total_investments_value += investment.quantity * todays_data['Close'][0]
            elif investment.asset_type == 'Crypto':
                price_data = cg.get_price(ids=investment.ticker_symbol, vs_currencies='usd')
                if price_data and price_data.get(investment.ticker_symbol):
                    current_price_usd = price_data[investment.ticker_symbol].get('usd', 0)
                    total_investments_value += (investment.quantity * current_price_usd) * usd_to_inr_rate
        except Exception as e:
            print(f"Net Worth: Could not fetch price for {investment.ticker_symbol}: {e}")
    total_assets = cash_balance + total_schemes_value + total_investments_value
    user_loans = Loan.query.filter_by(user_id=current_user.id).all()
    total_liabilities = 0
    loans_with_details = []
    for loan in user_loans:
        r = (loan.interest_rate / 12) / 100
        n = loan.tenure_months
        payments_made = (date.today().year - loan.start_date.year) * 12 + (date.today().month - loan.start_date.month)
        if payments_made < 0: payments_made = 0
        if payments_made > n: payments_made = n
        outstanding_balance = loan.principal * (((1 + r)**n) - ((1 + r)**payments_made)) / (((1 + r)**n) - 1) if (((1 + r)**n) - 1) != 0 else 0
        total_liabilities += outstanding_balance
        loans_with_details.append({'loan_name': loan.loan_name, 'outstanding': outstanding_balance})
    total_net_worth = total_assets - total_liabilities
    chart_data = {'labels': ['Cash', 'Fixed Schemes', 'Investments'], 'data': [cash_balance, total_schemes_value, total_investments_value]}
    return render_template('net_worth.html', 
                           net_worth=total_net_worth, 
                           assets=total_assets,
                           liabilities=total_liabilities,
                           cash=cash_balance, 
                           schemes=total_schemes_value, 
                           investments=total_investments_value,
                           loans=loans_with_details,
                           chart_data=json.dumps(chart_data))

@app.route('/tax_estimator')
@login_required
def tax_estimator():
    age = 0
    if current_user.dob:
        today = date.today()
        age = today.year - current_user.dob.year - ((today.month, today.day) < (current_user.dob.month, current_user.dob.day))
    salary_details = Salary.query.filter_by(user_id=current_user.id).first()
    gross_salary_income = 0
    deductions = 0
    if salary_details:
        gross_salary_income = salary_details.monthly_gross * 12
        deductions = salary_details.deductions_80c + salary_details.hra_exemption
    user_schemes = FixedScheme.query.filter_by(user_id=current_user.id).all()
    total_interest_income = 0
    for scheme in user_schemes:
        years_elapsed = (date.today() - scheme.start_date).days / 365.25
        if years_elapsed > 0:
            interest = scheme.principal_amount * ((1 + (scheme.interest_rate / 100)) ** years_elapsed) - scheme.principal_amount
            total_interest_income += interest
    sold_investments = SoldInvestment.query.filter_by(user_id=current_user.id).all()
    stcg_stocks = sum(s.capital_gain for s in sold_investments if s.asset_type == 'Stock' and s.gain_type == 'STCG')
    ltcg_stocks = sum(s.capital_gain for s in sold_investments if s.asset_type == 'Stock' and s.gain_type == 'LTCG')
    crypto_gains = sum(s.capital_gain for s in sold_investments if s.asset_type == 'Crypto')
    stcg_tax = stcg_stocks * 0.15
    ltcg_taxable = max(0, ltcg_stocks - 100000)
    ltcg_tax = ltcg_taxable * 0.10
    crypto_tax = crypto_gains * 0.30
    total_capital_gains_tax = stcg_tax + ltcg_tax + crypto_tax
    total_regular_income = gross_salary_income + total_interest_income
    new_regime_details = calculate_new_regime_tax(total_regular_income)
    old_regime_details = calculate_old_regime_tax(total_regular_income, deductions, age)
    new_regime_details['total_tax'] += total_capital_gains_tax
    old_regime_details['total_tax'] += total_capital_gains_tax
    capital_gains_summary = {
        'stcg_stocks': stcg_stocks, 'stcg_tax': stcg_tax,
        'ltcg_stocks': ltcg_stocks, 'ltcg_tax': ltcg_tax,
        'crypto_gains': crypto_gains, 'crypto_tax': crypto_tax,
        'total_tax': total_capital_gains_tax
    }
    return render_template('tax_estimator.html', 
                           new_regime=new_regime_details, 
                           old_regime=old_regime_details, 
                           salary_setup=(salary_details is not None), 
                           user_age=age,
                           capital_gains=capital_gains_summary,
                           interest_income=total_interest_income)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        dob_str = request.form.get('dob')
        if dob_str:
            current_user.dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
            db.session.commit()
            flash('Your profile has been updated.', 'success')
            return redirect(url_for('profile'))
    return render_template('profile.html', user=current_user)

@app.route('/sell_investment/<int:investment_id>', methods=['POST'])
@login_required
def sell_investment(investment_id):
    investment = db.session.get(Investment, investment_id)
    if not investment or investment.user_id != current_user.id:
        flash('Investment not found or not authorized.', 'error')
        return redirect(url_for('investments'))

    sell_price = float(request.form.get('sell_price'))
    sell_quantity = float(request.form.get('sell_quantity'))
    sell_date = datetime.strptime(request.form.get('sell_date'), '%Y-%m-%d').date()

    if sell_quantity > investment.quantity or sell_quantity <= 0:
        flash('Invalid quantity to sell.', 'error')
        return redirect(url_for('investments'))

    holding_period_days = (sell_date - investment.purchase_date).days
    
    gain_type = 'STCG'
    if investment.asset_type == 'Stock' and holding_period_days > 365:
        gain_type = 'LTCG'

    purchase_cost = investment.purchase_price * sell_quantity
    sell_value = sell_price * sell_quantity
    capital_gain = sell_value - purchase_cost
    
    new_sale = SoldInvestment(
        asset_type=investment.asset_type,
        ticker_symbol=investment.ticker_symbol,
        quantity=sell_quantity,
        purchase_price=investment.purchase_price,
        purchase_date=investment.purchase_date,
        sell_price=sell_price,
        sell_date=sell_date,
        capital_gain=capital_gain,
        gain_type=gain_type,
        user_id=current_user.id
    )
    db.session.add(new_sale)

    investment.quantity -= sell_quantity
    if investment.quantity <= 0.000001:
        db.session.delete(investment)
    
    db.session.commit()
    flash(f'Successfully sold {sell_quantity} units of {investment.ticker_symbol.upper()}.', 'success')
    return redirect(url_for('investments'))

@app.route('/loans')
@login_required
def loans():
    user_loans = Loan.query.filter_by(user_id=current_user.id).all()
    return render_template('loans.html', loans=user_loans)

@app.route('/add_loan', methods=['POST'])
@login_required
def add_loan():
    loan_name = request.form.get('loan_name')
    principal = float(request.form.get('principal'))
    rate = float(request.form.get('interest_rate'))
    tenure = int(request.form.get('tenure_months'))
    start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d').date()

    r = (rate / 12) / 100
    emi = (principal * r * (1 + r)**tenure) / ((1 + r)**tenure - 1)

    new_loan = Loan(
        loan_name=loan_name,
        principal=principal,
        interest_rate=rate,
        tenure_months=tenure,
        emi_amount=emi,
        start_date=start_date,
        user_id=current_user.id
    )
    db.session.add(new_loan)
    db.session.commit()
    flash(f'Loan "{loan_name}" added successfully with a calculated EMI of ₹{emi:.2f}.', 'success')
    return redirect(url_for('loans'))

@app.route('/delete_loan/<int:loan_id>', methods=['POST'])
@login_required
def delete_loan(loan_id):
    loan = db.session.get(Loan, loan_id)
    if not loan or loan.user_id != current_user.id:
        flash('Loan not found or not authorized.', 'error')
        return redirect(url_for('loans'))
    db.session.delete(loan)
    db.session.commit()
    flash('Loan deleted successfully.', 'success')
    return redirect(url_for('loans'))

@app.route('/sold_investments')
@login_required
def sold_investments():
    sales = SoldInvestment.query.filter_by(user_id=current_user.id).order_by(SoldInvestment.sell_date.desc()).all()
    return render_template('sold_investments.html', sales=sales)

@app.route('/edit_transaction/<int:transaction_id>', methods=['GET', 'POST'])
@login_required
def edit_transaction(transaction_id):
    transaction = db.session.get(Transaction, transaction_id)
    if not transaction or transaction.user_id != current_user.id:
        flash('Transaction not found or not authorized.', 'error')
        return redirect(url_for('view_transactions'))
    
    if request.method == 'POST':
        transaction.description = request.form.get('description')
        transaction.amount = float(request.form.get('amount'))
        transaction.type = request.form.get('type')
        transaction.category = request.form.get('category')
        transaction.date = datetime.strptime(request.form.get('date'), '%Y-%m-%d').date()
        db.session.commit()
        flash('Transaction updated successfully!', 'success')
        return redirect(url_for('view_transactions'))
        
    return render_template('edit_transaction.html', transaction=transaction)

@app.route('/edit_scheme/<int:scheme_id>', methods=['GET', 'POST'])
@login_required
def edit_scheme(scheme_id):
    scheme = db.session.get(FixedScheme, scheme_id)
    if not scheme or scheme.user_id != current_user.id:
        flash('Scheme not found or not authorized.', 'error')
        return redirect(url_for('schemes'))

    if request.method == 'POST':
        scheme.scheme_name = request.form.get('scheme_name')
        scheme.principal_amount = float(request.form.get('principal_amount'))
        scheme.interest_rate = float(request.form.get('interest_rate'))
        scheme.tenure_months = int(request.form.get('tenure_months'))
        scheme.start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d').date()
        scheme.penalty_rate = float(request.form.get('penalty_rate'))
        db.session.commit()
        flash('Scheme updated successfully!', 'success')
        return redirect(url_for('schemes'))

    return render_template('edit_scheme.html', scheme=scheme)

@app.route('/edit_loan/<int:loan_id>', methods=['GET', 'POST'])
@login_required
def edit_loan(loan_id):
    loan = db.session.get(Loan, loan_id)
    if not loan or loan.user_id != current_user.id:
        flash('Loan not found or not authorized.', 'error')
        return redirect(url_for('loans'))

    if request.method == 'POST':
        loan.loan_name = request.form.get('loan_name')
        loan.principal = float(request.form.get('principal'))
        loan.interest_rate = float(request.form.get('interest_rate'))
        loan.tenure_months = int(request.form.get('tenure_months'))
        loan.start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d').date()
        
        # Recalculate EMI
        r = (loan.interest_rate / 12) / 100
        tenure = loan.tenure_months
        loan.emi_amount = (loan.principal * r * (1 + r)**tenure) / ((1 + r)**tenure - 1)
        
        db.session.commit()
        flash('Loan updated successfully!', 'success')
        return redirect(url_for('loans'))

    return render_template('edit_loan.html', loan=loan)

if __name__ == '__main__':
    app.run(debug=True)
