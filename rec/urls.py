from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('recommend/', views.recommendMovie, name='recommend'),
    path('person/', views.queryPerson, name='person'),
    path('movie/', views.queryMovie, name='movie'),
    path('search/', views.searchMovie, name='search'),
]