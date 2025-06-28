"""
User Authentication and Authorization for Telegram Bot.

This module handles user whitelisting, rate limiting, and access control
for the Telegram bot interface.
"""
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque

from telegram import Update
from telegram.ext import ContextTypes

from .config import BotConfig

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    """Represents a user session with rate limiting and activity tracking."""
    user_id: int
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_admin: bool = False
    
    # Rate limiting
    command_timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    last_command_time: Optional[datetime] = None
    commands_this_minute: int = 0
    
    # Activity tracking
    session_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_commands: int = 0
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Command-specific cooldowns
    last_opportunities_time: Optional[datetime] = None
    last_status_time: Optional[datetime] = None
    
    def update_activity(self, command: str) -> None:
        """Update user activity tracking."""
        now = datetime.now(timezone.utc)
        self.last_activity = now
        self.last_command_time = now
        self.total_commands += 1
        
        # Add to command timestamps for rate limiting
        self.command_timestamps.append(now)
        
        # Count commands in the last minute
        minute_ago = now - timedelta(minutes=1)
        self.commands_this_minute = sum(
            1 for ts in self.command_timestamps 
            if ts > minute_ago
        )
        
        # Update command-specific timestamps
        if command == 'opportunities':
            self.last_opportunities_time = now
        elif command == 'status':
            self.last_status_time = now
    
    def is_rate_limited(self, config: BotConfig, command: str) -> bool:
        """Check if user is rate limited for a specific command."""
        now = datetime.now(timezone.utc)
        
        # Check general rate limit
        if self.commands_this_minute >= config.commands_per_minute:
            return True
        
        # Check command-specific cooldowns
        if command == 'opportunities' and self.last_opportunities_time:
            cooldown = timedelta(seconds=config.opportunities_cooldown_seconds)
            if now - self.last_opportunities_time < cooldown:
                return True
        
        if command == 'status' and self.last_status_time:
            cooldown = timedelta(seconds=config.status_cooldown_seconds)
            if now - self.last_status_time < cooldown:
                return True
        
        return False
    
    def get_cooldown_remaining(self, config: BotConfig, command: str) -> Optional[int]:
        """Get remaining cooldown time in seconds for a command."""
        now = datetime.now(timezone.utc)
        
        if command == 'opportunities' and self.last_opportunities_time:
            cooldown = timedelta(seconds=config.opportunities_cooldown_seconds)
            remaining = cooldown - (now - self.last_opportunities_time)
            if remaining.total_seconds() > 0:
                return int(remaining.total_seconds())
        
        if command == 'status' and self.last_status_time:
            cooldown = timedelta(seconds=config.status_cooldown_seconds)
            remaining = cooldown - (now - self.last_status_time)
            if remaining.total_seconds() > 0:
                return int(remaining.total_seconds())
        
        return None


class UserAuthenticator:
    """Handles user authentication, authorization, and session management."""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.sessions: Dict[int, UserSession] = {}
        self.blocked_users: set = set()
        self.failed_auth_attempts: Dict[int, List[datetime]] = defaultdict(list)
        
        logger.info(f"UserAuthenticator initialized with {len(config.allowed_user_ids)} allowed users")
    
    def authenticate_user(self, update: Update) -> Optional[UserSession]:
        """Authenticate a user and return their session."""
        if not update.effective_user:
            return None
        
        user = update.effective_user
        user_id = user.id
        
        # Check if user is blocked
        if user_id in self.blocked_users:
            logger.warning(f"Blocked user {user_id} attempted to use bot")
            return None
        
        # Check if user is allowed
        if not self.config.is_user_allowed(user_id):
            self._record_failed_auth(user_id)
            logger.warning(f"Unauthorized user {user_id} ({user.username}) attempted to use bot")
            return None
        
        # Get or create user session
        if user_id not in self.sessions:
            session = UserSession(
                user_id=user_id,
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name,
                is_admin=self.config.is_user_admin(user_id)
            )
            self.sessions[user_id] = session
            logger.info(f"Created new session for user {user_id} ({user.username})")
        else:
            session = self.sessions[user_id]
        
        return session
    
    def check_rate_limit(self, session: UserSession, command: str) -> bool:
        """Check if user is rate limited for a command."""
        return session.is_rate_limited(self.config, command)
    
    def update_user_activity(self, session: UserSession, command: str) -> None:
        """Update user activity tracking."""
        session.update_activity(command)
    
    def _record_failed_auth(self, user_id: int) -> None:
        """Record a failed authentication attempt."""
        now = datetime.now(timezone.utc)
        self.failed_auth_attempts[user_id].append(now)
        
        # Clean old attempts (older than 1 hour)
        hour_ago = now - timedelta(hours=1)
        self.failed_auth_attempts[user_id] = [
            ts for ts in self.failed_auth_attempts[user_id] if ts > hour_ago
        ]
        
        # Block user if too many failed attempts
        if len(self.failed_auth_attempts[user_id]) >= 5:
            self.blocked_users.add(user_id)
            logger.warning(f"Blocked user {user_id} due to repeated unauthorized access attempts")
    
    def block_user(self, user_id: int) -> None:
        """Manually block a user."""
        self.blocked_users.add(user_id)
        if user_id in self.sessions:
            del self.sessions[user_id]
        logger.info(f"Manually blocked user {user_id}")
    
    def unblock_user(self, user_id: int) -> None:
        """Unblock a user."""
        self.blocked_users.discard(user_id)
        if user_id in self.failed_auth_attempts:
            del self.failed_auth_attempts[user_id]
        logger.info(f"Unblocked user {user_id}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions."""
        now = datetime.now(timezone.utc)
        active_sessions = 0
        total_commands = 0
        
        for session in self.sessions.values():
            if session.last_activity and (now - session.last_activity).total_seconds() < 3600:
                active_sessions += 1
            total_commands += session.total_commands
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "blocked_users": len(self.blocked_users),
            "total_commands_processed": total_commands,
            "failed_auth_attempts": sum(len(attempts) for attempts in self.failed_auth_attempts.values())
        }
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old inactive sessions."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=max_age_hours)
        
        old_sessions = [
            user_id for user_id, session in self.sessions.items()
            if session.last_activity < cutoff
        ]
        
        for user_id in old_sessions:
            del self.sessions[user_id]
        
        logger.info(f"Cleaned up {len(old_sessions)} old sessions")
        return len(old_sessions)


def auth_required(func: Callable) -> Callable:
    """Decorator to require authentication for command handlers."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        if not hasattr(context.bot_data, 'authenticator'):
            await update.message.reply_text("⚠️ Authentication system not available")
            return
        
        authenticator: UserAuthenticator = context.bot_data['authenticator']
        session = authenticator.authenticate_user(update)
        
        if not session:
            await update.message.reply_text(
                "❌ Access denied. You are not authorized to use this bot."
            )
            return
        
        # Extract command name from function name or update
        command_name = func.__name__.replace('_command', '').replace('_handler', '')
        
        # Check rate limiting
        if authenticator.check_rate_limit(session, command_name):
            cooldown = session.get_cooldown_remaining(authenticator.config, command_name)
            if cooldown:
                await update.message.reply_text(
                    f"⏱️ Command on cooldown. Please wait {cooldown} seconds."
                )
            else:
                await update.message.reply_text(
                    "⏱️ Rate limit exceeded. Please wait a moment before trying again."
                )
            return
        
        # Update activity and proceed
        authenticator.update_user_activity(session, command_name)
        context.user_data['session'] = session
        
        return await func(update, context, *args, **kwargs)
    
    return wrapper


def admin_required(func: Callable) -> Callable:
    """Decorator to require admin privileges for command handlers."""
    @auth_required
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        session: UserSession = context.user_data.get('session')
        
        if not session or not session.is_admin:
            await update.message.reply_text(
                "❌ Admin privileges required for this command."
            )
            return
        
        return await func(update, context, *args, **kwargs)
    
    return wrapper