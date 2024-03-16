from django.urls import path
from . import views

urlpatterns = [
    path('save-json', views.saveJson, name='save_json'),
    path('excel-export', views.excel_export, name='excel-export'),
    path('matrix', views.matrix, name='matrix'),
]