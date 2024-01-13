# image_classification_app/urls.py
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import HomeView, TestModelView, ResultView, ModelPerformanceView, ModelSummaryView, ModelSummaryDetailView, ModelPerformanceDetailView

urlpatterns = [
    path('', HomeView.as_view(), name='home'),
    path('test_model/', TestModelView.as_view(), name='test_model'),
    path('result/', ResultView.as_view(), name='result'),
    path('model_performance/', ModelPerformanceView.as_view(), name='model_performance'),
    path('model_performance/<str:model_name>_model_performance/', ModelPerformanceDetailView.as_view(), name='model_performance_detail'),
    path('model_summary/', ModelSummaryView.as_view(), name='model_summary'),
    path('model_summary/<str:model_name>_model_summary/', ModelSummaryDetailView.as_view(), name='model_summary_detail'),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
