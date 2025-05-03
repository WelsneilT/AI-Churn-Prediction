# churn_app/urls.py
from django.urls import path
from .views import predict_online, predict_batch, dashboard_view # Import 2 views mới

app_name = 'churn_app'

urlpatterns = [
    # Trang chủ và Online Prediction trỏ về cùng view predict_online
    path('', predict_online, name='home'),
    path('predict/online/', predict_online, name='predict_online'),

    # Đường dẫn cho Batch Prediction
    path('predict/batch/', predict_batch, name='predict_batch'),
    path('dashboard/', dashboard_view, name='dashboard'),


]