"""
Django admin registration for accounts.

The stock ``UserAdmin`` assumes a ``username`` field and Django's default
password widgets, so this app registers a slimmed-down ``ModelAdmin`` built
around the e-mail login instead.  Password hashes are never rendered or
editable here — resets go through the API's password-reset flow or
``manage.py changepassword``.
"""
from django.contrib import admin

from .models import PasswordResetToken, User


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    """Read-mostly view of the user table."""

    list_display = ('email', 'full_name', 'role', 'is_active', 'is_staff',
                    'created_at', 'last_login')
    list_filter = ('role', 'is_active', 'is_staff', 'is_superuser', 'created_at')
    search_fields = ('email', 'full_name', 'user_id')
    ordering = ('-created_at',)
    readonly_fields = ('user_id', 'created_at', 'last_login', 'password')
    filter_horizontal = ('groups', 'user_permissions')
    fieldsets = (
        (None, {'fields': ('user_id', 'email', 'full_name', 'role')}),
        ('Status', {'fields': ('is_active', 'is_staff', 'is_superuser')}),
        ('Permissions', {
            'classes': ('collapse',),
            'fields': ('groups', 'user_permissions'),
        }),
        ('Timestamps', {'fields': ('created_at', 'last_login')}),
        ('Credentials', {
            'classes': ('collapse',),
            'description': 'Hash only — use the password-reset flow to change '
                           'a password.',
            'fields': ('password',),
        }),
    )


@admin.register(PasswordResetToken)
class PasswordResetTokenAdmin(admin.ModelAdmin):
    """Audit view of outstanding password-reset links."""

    list_display = ('token_id', 'user', 'created_at', 'expires_at', 'used',
                    'valid_now')
    list_filter = ('used', 'created_at', 'expires_at')
    search_fields = ('user__email', 'token', 'token_id')
    ordering = ('-created_at',)
    readonly_fields = ('token_id', 'token', 'created_at')
    autocomplete_fields = ('user',)

    @admin.display(boolean=True, description='Currently valid')
    def valid_now(self, obj):
        return obj.is_valid()
