# Staging and Deployment Guide

## 🚀 **Staging Environment Setup**

### Option 1: Local Development (Free)
```bash
# Run locally for testing
python app.py
# Access at http://localhost:5000
```

### Option 2: Separate Staging Domain
- Create a staging subdomain: `staging.vallionis-ai-finance.onrender.com`
- Deploy changes there first
- Test thoroughly before pushing to production

### Option 3: Git Branches
```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and test locally
python app.py

# Merge to main only when ready
git checkout main
git merge feature/new-feature
```

## 🔧 **Development Workflow**

### **Step 1: Local Development**
1. Make changes locally
2. Test thoroughly on `localhost:5000`
3. Use browser dev tools to check everything

### **Step 2: Staging Deployment**
1. Deploy to staging environment
2. Test all functionality
3. Check for any issues

### **Step 3: Production Deployment**
1. Only deploy when 100% confident
2. Deploy during low-traffic hours
3. Monitor immediately after deployment

## 🛡️ **Protecting Your SEO During Updates**

### **1. Use Feature Flags**
```python
# In your app.py
FEATURE_FLAGS = {
    'new_design': False,  # Toggle new features
    'beta_features': False
}

@app.route('/')
def index():
    if FEATURE_FLAGS['new_design']:
        return render_template('index_new.html')
    else:
        return render_template('index.html')
```

### **2. Gradual Rollouts**
- Deploy to 10% of users first
- Monitor performance and errors
- Gradually increase to 100%

### **3. A/B Testing**
- Show different versions to different users
- Measure which performs better
- Keep the winning version

## 📝 **Content Update Strategies**

### **1. Draft Mode**
```python
@app.route('/admin/draft/<page>')
@login_required
def draft_page(page):
    # Show draft content to admins only
    if current_user.is_admin:
        return render_template(f'draft_{page}.html')
    return redirect(url_for('index'))
```

### **2. Scheduled Updates**
```python
from datetime import datetime

def should_show_new_content():
    # Only show new content after specific date
    return datetime.now() > datetime(2025, 8, 1)

@app.route('/')
def index():
    if should_show_new_content():
        return render_template('index_new.html')
    return render_template('index.html')
```

### **3. User-Specific Content**
```python
@app.route('/')
def index():
    # Show new design only to logged-in users
    if current_user.is_authenticated:
        return render_template('index_new.html')
    return render_template('index.html')
```

## 🔄 **Database Migration Strategies**

### **1. Backward Compatibility**
```python
# Keep old and new database schemas
class User(db.Model):
    # Old fields
    email = db.Column(db.String(120))
    
    # New fields (optional)
    phone = db.Column(db.String(20), nullable=True)
    
    def get_phone(self):
        return self.phone or "Not provided"
```

### **2. Database Versioning**
```python
# Check database version
def get_db_version():
    return db.session.execute("SELECT version FROM schema_version").scalar()

def migrate_if_needed():
    current_version = get_db_version()
    if current_version < 2:
        # Run migration
        db.session.execute("ALTER TABLE users ADD COLUMN phone VARCHAR(20)")
        db.session.commit()
```

## 🚨 **Emergency Rollback Plan**

### **1. Git Rollback**
```bash
# If something goes wrong
git log --oneline  # Find the last good commit
git reset --hard <commit-hash>
git push --force origin main
```

### **2. Database Rollback**
```python
# Keep backup before major changes
def backup_database():
    # Create backup before changes
    pass

def rollback_database():
    # Restore from backup if needed
    pass
```

### **3. Feature Toggle Rollback**
```python
# Quick rollback using environment variables
import os

FEATURE_FLAGS = {
    'new_design': os.getenv('NEW_DESIGN', 'false').lower() == 'true'
}
```

## 📊 **Monitoring During Updates**

### **1. Health Checks**
```python
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'database': check_database(),
        'features': FEATURE_FLAGS
    })
```

### **2. Error Monitoring**
```python
@app.errorhandler(500)
def internal_error(error):
    # Log errors and send alerts
    logger.error(f"Server error: {error}")
    # Send notification to admin
    return render_template('500.html'), 500
```

### **3. Performance Monitoring**
```python
from time import time

@app.before_request
def start_timer():
    g.start = time()

@app.after_request
def log_request(response):
    if hasattr(g, 'start'):
        duration = time() - g.start
        logger.info(f"Request to {request.path} took {duration:.2f}s")
    return response
```

## 🎯 **Best Practices**

### **Before Deployment:**
1. ✅ Test locally
2. ✅ Test on staging
3. ✅ Backup database
4. ✅ Check all features work
5. ✅ Monitor error logs

### **During Deployment:**
1. ✅ Deploy during low traffic
2. ✅ Monitor health checks
3. ✅ Watch error rates
4. ✅ Check user feedback

### **After Deployment:**
1. ✅ Monitor performance
2. ✅ Check Google Search Console
3. ✅ Verify all features work
4. ✅ Monitor user behavior

## 🔧 **Environment Variables for Control**

```bash
# .env file
NEW_DESIGN=false
BETA_FEATURES=false
MAINTENANCE_MODE=false
```

```python
# In app.py
NEW_DESIGN = os.getenv('NEW_DESIGN', 'false').lower() == 'true'
BETA_FEATURES = os.getenv('BETA_FEATURES', 'false').lower() == 'true'
MAINTENANCE_MODE = os.getenv('MAINTENANCE_MODE', 'false').lower() == 'true'
```

## 📞 **Quick Commands**

### **Deploy to Staging:**
```bash
git checkout staging
git merge main
git push origin staging
```

### **Deploy to Production:**
```bash
git checkout main
git merge staging
git push origin main
```

### **Rollback:**
```bash
git reset --hard HEAD~1
git push --force origin main
``` 