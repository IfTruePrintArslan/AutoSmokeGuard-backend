"""
Add ``accounts.User.password_changed_at`` — security review finding ASG-08.

A password reset blacklists every refresh token but could not touch an access
token that had already been minted, leaving an attacker with full API access
for up to the token's remaining 30-minute lifetime.  This column is the single
piece of per-user server-side state that
``accounts.authentication.PasswordChangeAwareJWTAuthentication`` compares an
access token's ``iat`` claim against.

**Deliberately not backfilled.**  The column ships NULL for every existing row,
and NULL means "no password change has ever been recorded" — i.e. no
restriction — not "reject everything".  Backfilling it to ``now()`` would sign
out every currently authenticated user the moment this migration ran, for no
security benefit: nobody's password actually changed.  Rows start carrying a
value the first time each account's password is set
(``accounts.models.User.set_password``), which is exactly when the guarantee
becomes meaningful.

Purely additive and nullable, so it applies to an existing SQLite or
PostgreSQL database without a table rewrite or a default backfill, and is
reversible with no data loss.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='password_changed_at',
            field=models.DateTimeField(
                null=True,
                blank=True,
                editable=False,
                help_text=(
                    'When this password was last set. Access tokens issued '
                    'before this moment are refused — see '
                    'accounts.authentication.'
                ),
            ),
        ),
    ]
