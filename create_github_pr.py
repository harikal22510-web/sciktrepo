import os
from github import Github
from github.Auth import Token

def create_github_pr():
    # Try to get token from environment
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print('❌ GitHub token not found in environment variables')
        print('Please set GITHUB_TOKEN environment variable')
        return False
    
    try:
        # Initialize GitHub client
        g = Github(auth=Token(token))
        repo = g.get_repo('sandeeptridizi/newpipemed')
        
        # Create the PR
        pr = repo.create_pull(
            title='Fix code quality issues: modernize JavaScript and remove TODO comments',
            body='''## Summary
This PR addresses several code quality issues found in the repository:

### Changes Made
- **JavaScript Modernization**: Converted `var` declarations to `let/const` in `main.js` for modern JavaScript best practices
- **TODO Cleanup**: Removed unnecessary TODO comments from Python files that were not actionable
- **Code Formatting**: Applied `black` code formatting to improve consistency

### Files Modified
- `mediapipe/examples/desktop/youtube8m/viewer/static/main.js`
- `mediapipe/python/solution_base.py` 
- `mediapipe/python/solutions/drawing_utils.py`

### Impact
- Improved code maintainability
- Better adherence to modern JavaScript standards
- Cleaner codebase with removed technical debt
- Consistent code formatting

### Testing
- Changes are purely cosmetic/code quality improvements
- No functional changes to existing behavior
- All existing functionality preserved

This PR helps improve overall code quality and maintainability of the codebase.''',
            head='five-file-fix',
            base='main'
        )
        
        print(f'✅ PR created successfully!')
        print(f'🔗 PR URL: {pr.html_url}')
        print(f'📝 PR #{pr.number}: {pr.title}')
        return True
        
    except Exception as e:
        print(f'❌ Error creating PR: {e}')
        return False

if __name__ == "__main__":
    create_github_pr()
