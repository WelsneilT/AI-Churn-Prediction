# churn_app/forms.py
from django import forms

# Form Upload CSV (giữ nguyên)
class UploadFileForm(forms.Form):
    csv_file = forms.FileField(
        label='Select CSV file',
        help_text='Upload a CSV file with customer data for batch prediction.'
    )

# --- Form MỚI cho Online Prediction ---
class OnlinePredictionForm(forms.Form):
    # --- Demographic Data ---
    gender_choices = [('Male', 'Male'), ('Female', 'Female')]
    gender = forms.ChoiceField(choices=gender_choices, label="Gender") # Giữ nguyên ChoiceField vì không coerce

    yes_no_choices = [(1, 'Yes'), (0, 'No')] # Giá trị 1/0

    # SỬ DỤNG TypedChoiceField KHI CÓ coerce
    senior_citizen = forms.TypedChoiceField(choices=yes_no_choices, label="Senior Citizen", widget=forms.Select, coerce=int)
    partner = forms.TypedChoiceField(choices=yes_no_choices, label="Partner", widget=forms.Select, coerce=int)
    dependents = forms.TypedChoiceField(choices=yes_no_choices, label="Dependents", widget=forms.Select, coerce=int)

    age = forms.IntegerField(min_value=18, max_value=100, label="Age")

    # --- Account Information ---
    tenure = forms.IntegerField(min_value=0, label="Tenure (months)")

    contract_choices = [('Month-to-month', 'Month-to-month'), ('One year', 'One year'), ('Two year', 'Two year')]
    contract = forms.ChoiceField(choices=contract_choices, label="Contract") # Giữ nguyên ChoiceField

    # SỬ DỤNG TypedChoiceField KHI CÓ coerce
    paperless_billing = forms.TypedChoiceField(choices=yes_no_choices, label="Paperless Billing", widget=forms.Select, coerce=int)

    payment_method_choices = [
        ('Electronic check', 'Electronic check'),
        ('Mailed check', 'Mailed check'),
        ('Bank transfer (automatic)', 'Bank transfer (automatic)'),
        ('Credit card (automatic)', 'Credit card (automatic)')
    ]
    payment_method = forms.ChoiceField(choices=payment_method_choices, label="Payment Method") # Giữ nguyên ChoiceField

    monthly_charges = forms.FloatField(min_value=0, label="Monthly Charges")
    total_charges = forms.FloatField(min_value=0, label="Total Charges")

    # --- Services ---
    # SỬ DỤNG TypedChoiceField KHI CÓ coerce
    phone_service = forms.TypedChoiceField(choices=yes_no_choices, label="Phone Service", widget=forms.Select, coerce=int)

    # Giữ nguyên ChoiceField vì giá trị có thể là text 'No phone service'
    multiple_lines = forms.ChoiceField(
        choices=[('Yes', 'Yes'), ('No', 'No'), ('No phone service', 'No phone service')], # Giá trị gốc có thể là text
        label="Multiple Lines",
        widget=forms.Select
    )
    # Giữ nguyên ChoiceField vì giá trị là text
    internet_service = forms.ChoiceField(
        choices=[('DSL', 'DSL'), ('Fiber optic', 'Fiber optic'), ('No', 'No internet service')],
        label="Internet Service"
    )

    # Giữ nguyên ChoiceField vì giá trị có thể là text 'No internet service'
    online_security_choices = [('Yes', 'Yes'), ('No', 'No'), ('No internet service', 'No internet service')]
    online_security = forms.ChoiceField(choices=online_security_choices, label="Online Security", widget=forms.Select)
    online_backup = forms.ChoiceField(choices=online_security_choices, label="Online Backup", widget=forms.Select)
    device_protection = forms.ChoiceField(choices=online_security_choices, label="Device Protection", widget=forms.Select)
    tech_support = forms.ChoiceField(choices=online_security_choices, label="Tech Support", widget=forms.Select) # Đổi tên biến từ tech_support thành khác nếu trùng
    streaming_tv = forms.ChoiceField(choices=online_security_choices, label="Streaming TV", widget=forms.Select)
    streaming_movies = forms.ChoiceField(choices=online_security_choices, label="Streaming Movies", widget=forms.Select)

    # ... (các trường khác nếu có) ...

    def clean(self):
        # ... (phần clean giữ nguyên) ...
        cleaned_data = super().clean()
        internet = cleaned_data.get('internet_service')
        internet_options = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
        if internet == 'No':
            for option in internet_options:
                # Kiểm tra giá trị gốc trước khi gán lại
                if cleaned_data.get(option) != 'No internet service':
                     cleaned_data[option] = 'No internet service'

        phone = cleaned_data.get('phone_service') # Bây giờ phone là 0 hoặc 1
        if phone == 0:
             if cleaned_data.get('multiple_lines') != 'No phone service':
                  cleaned_data['multiple_lines'] = 'No phone service'

        return cleaned_data