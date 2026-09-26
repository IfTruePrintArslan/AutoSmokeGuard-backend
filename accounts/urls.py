"""
URL routes for the accounts API, mounted under ``/api/`` by ``config.urls``.

Paths match the frozen API contract exactly — flat, verb-specific, and
without a trailing slash — so no route here should ever be renamed or
nested under a router.
"""
from django.urls import path

from . import views

urlpatterns = [
    path('register', views.RegisterView.as_view(), name='accounts-register'),
    path('login', views.LoginView.as_view(), name='accounts-login'),
    path('refresh-token', views.RefreshTokenView.as_view(),
         name='accounts-refresh-token'),
    path('logout', views.LogoutView.as_view(), name='accounts-logout'),
    path('me', views.MeView.as_view(), name='accounts-me'),
    path('password-reset', views.PasswordResetRequestView.as_view(),
         name='accounts-password-reset'),
    path('password-reset/confirm', views.PasswordResetConfirmView.as_view(),
         name='accounts-password-reset-confirm'),
]
