# Security Policy

## Overview

The security of SHAP Analytics is a top priority. We appreciate the efforts of security researchers and users who help us maintain a secure platform for ML explainability.

This document outlines our security policy, including supported versions, how to report vulnerabilities, and security best practices.


## Supported Versions

We provide security updates for the following versions of SHAP Analytics:

| Version | Supported          | End of Support |
| ------- | ------------------ | -------------- |
| 1.x.x   | :white_check_mark: | TBD            |
| 0.9.x   | :white_check_mark: | 2025-12-31     |
| 0.8.x   | :x:                | 2024-06-30     |
| < 0.8   | :x:                | Ended          |

**Note:** We strongly recommend always using the latest stable version to ensure you have the most recent security patches and improvements.


## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability, please follow these steps:

### 1. Contact Us Privately

Send a detailed report to: **security@yourorganization.com** (or create a private security advisory on GitHub)

**Alternatively**, you can use [GitHub Security Advisories](https://github.com/yourusername/shap-analytics/security/advisories/new) to report vulnerabilities privately.

### 2. Include in Your Report

Please include the following information to help us understand and address the issue quickly:

- **Type of vulnerability** (e.g., code injection, authentication bypass, data exposure)
- **Affected component(s)** (specific files, endpoints, or features)
- **Attack scenario** (step-by-step reproduction)
- **Impact assessment** (what data or functionality is at risk)
- **Proof of concept** (code, screenshots, or detailed description)
- **Suggested fix** (if you have one)
- **Your contact information** for follow-up questions
- **Disclosure timeline preference** (when you plan to publicly disclose, if applicable)

### 3. What to Expect

| Timeline | Action |
| -------- | ------ |
| **Within 24 hours** | We will acknowledge receipt of your report |
| **Within 3 business days** | We will provide an initial assessment and timeline |
| **Within 7 business days** | We will confirm the vulnerability and begin remediation |
| **Within 30 days** | We aim to release a fix for critical vulnerabilities |
| **Within 90 days** | We aim to release a fix for medium/low severity issues |

**Note:** Complex vulnerabilities may require additional time. We will keep you informed throughout the process.


## Vulnerability Disclosure Policy

We follow a **coordinated disclosure** approach:

1. **Private Reporting:** You report the vulnerability privately to our security team
2. **Assessment & Fix:** We assess, develop, and test a fix
3. **Release:** We release a patched version
4. **Public Disclosure:** We publish a security advisory with details
5. **Credit:** We publicly acknowledge your contribution (if you wish)

We request that security researchers:
- Give us reasonable time to address the issue before public disclosure
- Make a good faith effort to avoid privacy violations and service disruption
- Do not exploit the vulnerability beyond what is necessary to demonstrate it


## Security Best Practices for Users

### For Application Security

1. **Keep Dependencies Updated**
   ```bash
   pip install --upgrade shap-analytics
   pip install --upgrade -r requirements.txt
   ```

2. **Use Virtual Environments**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Validate Input Data**
   - Always validate and sanitize input data before processing
   - Use type hints and validation libraries (e.g., Pydantic)
   - Implement rate limiting for API endpoints

4. **Secure Docker Deployments**
   - Run containers as non-root user
   - Keep Docker images updated
   - Use specific version tags, not `latest`
   - Scan images for vulnerabilities:
     ```bash
     docker scan shap-analytics:latest
     ```

5. **Environment Variables**
   - Never commit `.env` files with secrets
   - Use secret management systems (e.g., AWS Secrets Manager, HashiCorp Vault)
   - Rotate credentials regularly

6. **FastAPI Security**
   - Enable CORS only for trusted origins
   - Implement authentication and authorization
   - Use HTTPS in production
   - Set appropriate security headers

### For Data Security

1. **Sensitive Data Handling**
   - Avoid logging sensitive data (PII, credentials, etc.)
   - Sanitize data before generating SHAP explanations
   - Consider data anonymization techniques

2. **Model Security**
   - Validate model inputs to prevent adversarial attacks
   - Be cautious with user-supplied models (pickle files can execute arbitrary code)
   - Use signed and verified model artifacts

3. **Access Control**
   - Implement proper authentication for API endpoints
   - Use role-based access control (RBAC) where appropriate
   - Audit access to sensitive explanations


## Scope

### In Scope

The following are considered security vulnerabilities:

- **Code Injection:** SQL injection, command injection, code execution
- **Authentication/Authorization Bypass**
- **Data Exposure:** Unauthorized access to data or explanations
- **Denial of Service (DoS):** Resource exhaustion attacks
- **Dependency Vulnerabilities:** Critical vulnerabilities in dependencies
- **API Security Issues:** Unauthorized API access, rate limit bypass
- **Docker Security Issues:** Container escape, privilege escalation

### Out of Scope

The following are generally **not** considered security vulnerabilities:

- Issues in unsupported versions (see table above)
- Issues requiring physical access to the server
- Social engineering attacks
- Denial of service requiring massive resources
- Issues in third-party dependencies (report to the dependency maintainers)
- Bugs without security impact (report as regular bug reports)
- Performance issues without security implications


## Security Features

SHAP Analytics includes the following security features:

- **Input Validation:** Comprehensive validation of user inputs
- **Type Safety:** Extensive use of Python type hints
- **Dependency Scanning:** Regular automated scanning of dependencies
- **Docker Security:** Non-root containers, minimal base images
- **API Rate Limiting:** Protection against abuse (when configured)
- **Secure Defaults:** Security-first default configurations


## Security Updates

Security updates are released as:

- **Patch versions** (e.g., 1.0.1) for minor security fixes
- **Minor versions** (e.g., 1.1.0) for security features and moderate fixes
- **Major versions** (e.g., 2.0.0) for breaking security changes

Subscribe to our [security advisories](https://github.com/yourusername/shap-analytics/security/advisories) to receive notifications.


## Compliance

SHAP Analytics is designed to support compliance with:

- **GDPR:** Data minimization, right to explanation
- **HIPAA:** When deployed with appropriate safeguards
- **SOC 2:** Security controls and monitoring
- **ISO 27001:** Information security management

**Note:** Compliance is a shared responsibility. Users must implement appropriate controls for their specific use case.


## Security Acknowledgments

We would like to thank the following security researchers for responsibly disclosing vulnerabilities:

<!-- Will be updated as researchers contribute -->
- *No vulnerabilities reported yet*


## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)


## Contact

- **Security Email:** security@yourorganization.com
- **General Contact:** support@yourorganization.com
- **GitHub Security Advisories:** [Report a vulnerability](https://github.com/yourusername/shap-analytics/security/advisories/new)


## Policy Updates

This security policy is reviewed and updated quarterly. Last updated: 2024-11-07

---

**Thank you for helping keep SHAP Analytics and our users safe!**
