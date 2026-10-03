# NIST Special Publication 800-53 (Revision 5)
## Security and Privacy Controls for Information Systems and Organizations

### Control Family: Access Control (AC) & Identification and Authentication (IA)

---

## AC-1: Policy and Procedures

### Control Description
The organization:
a. Develops, documents, and disseminates to designated personnel:
   1. Organization-level access control policy that addresses purpose, scope, roles, responsibilities, management commitment, coordination among organizational entities, and compliance; and
   2. Procedures to facilitate the implementation of the access control policy and associated access controls; and
b. Reviews and updates the current:
   1. Access control policy at an organization-defined frequency; and
   2. Access control procedures at an organization-defined frequency.

### Control Baselines and Parameters
| Baseline Level | AC-1 Included | Review Frequency (Policy) | Review Frequency (Procedures) |
| :--- | :---: | :---: | :---: |
| **Low** | Yes | Annual | Annual |
| **Moderate** | Yes | Annual | Annual |
| **High** | Yes | Semi-Annual | Annual |

---

## AC-2: Account Management

### Control Description
The organization:
a. Identifies and selects the following types of information system accounts to support organizational missions/business functions: individual, guest/anonymous, temporary, service, and administrative;
b. Assigns account managers for information system accounts;
c. Establishes conditions for group and role membership;
d. Specifies authorized users, group and role membership, and access authorizations (i.e., privileges) and other attributes (as required) for each account;
e. Requires approvals by authorized personnel for requests to create information system accounts;
f. Creates, enables, modifies, disables, and removes information system accounts in accordance with organizational procedures;
g. Monitors the use of information system accounts;
h. Deactivates temporary and emergency accounts after thirty (30) days;
i. Disables inactive accounts after ninety (90) days; and
j. Notifies account managers when accounts are no longer required or when users are terminated.

### Control Enhancements

#### AC-2(1): Automated System Account Management
The organization employs automated mechanisms to support the management of information system accounts.
Automated tools synchronize account provisioning, role transitions, and de-provisioning with enterprise HR identity management databases.

```yaml
account_lifecycle_policy:
  auto_deprovisioning:
    enabled: true
    trigger: hr_termination_webhook
    grace_period_minutes: 0
    revoke_active_sessions: true
    archive_audit_trail_days: 365
```

#### AC-2(3): Inactive Account Disabling
The information system automatically disables inactive accounts after ninety (90) days of inactivity.
Privileged root/administrator accounts are subjected to a reduced threshold of thirty (30) days of inactivity before automatic suspension.

---

## AC-3: Access Enforcement

### Control Description
The information system enforces approved authorizations for logical access to information and system resources in accordance with applicable access control policies.
Access control policies control access between active subjects (e.g., human users, automated system processes) and passive objects (e.g., tables, files, database records, network sockets).

### Enforcement Mechanisms
Access control mechanisms can be implemented via:
1. **Attribute-Based Access Control (ABAC)**: Evaluates user attributes (department, clearance), environmental attributes (IP subnet, device posture, time of day), and resource sensitivity tags.
2. **Role-Based Access Control (RBAC)**: Maps permissions to organizational roles (e.g., Database Administrator, Compliance Auditor).

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EnforceMFAAndSubnetBoundary",
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "BoolIfExists": { "aws:MultiFactorAuthPresent": "false" },
        "NotIpAddress": { "aws:SourceIp": ["10.200.0.0/16", "192.168.1.0/24"] }
      }
    }
  ]
}
```

---

## AC-6: Least Privilege

### Control Description
The organization employs the principle of least privilege, allowing only authorized accesses for users (and processes acting on behalf of users) which are necessary to accomplish assigned tasks in accordance with organizational missions and business functions.

### Control Enhancements

#### AC-6(1): Authorize Access to Security Functions
The organization explicitly authorizes access to security-relevant functions and security-relevant information only to designated personnel.
Explicit authorizations include modifying audit logging configurations, adjusting firewall rules, generating cryptographic keys, and rotating database encryption secrets.

#### AC-6(2): Non-Privileged Access for Non-Security Functions
Privileged users utilize dedicated unprivileged accounts when performing non-administrative tasks (e.g., reading email, browsing the web).

#### AC-6(9): Auditing Use of Privileged Functions
The information system audits the execution of privileged functions and generates automated alerts for high-risk system commands (e.g., `sudo`, `DROP DATABASE`, kernel module insertion).

---

## IA-2: Identification and Authentication (Organizational Users)

### Control Description
The information system uniquely identifies and authenticates organizational users (or processes acting on behalf of organizational users).

### Control Enhancements

#### IA-2(1): Multi-Factor Authentication to Privileged Accounts
The information system implements phishing-resistant multi-factor authentication (MFA) for access to privileged accounts.
Hardware security keys (FIDO2 / WebAuthn) are mandatory for all production cloud infrastructure access. SMS-based and voice-based OTP authenticators are strictly prohibited for administrative access.

#### IA-2(2): Multi-Factor Authentication to Non-Privileged Accounts
The information system implements multi-factor authentication for access to non-privileged accounts across all web applications and internal network portals.

---

## IA-5: Authenticator Management

### Control Description
The organization manages information system authenticators by:
a. Defining initial authenticator content (e.g., minimum length 16 characters, entropy requirements);
b. Establishing administrative procedures for authenticator distribution, revocation, and recovery;
c. Changing default authenticators prior to system deployment; and
d. Enforcing cryptographically secure storage of passwords using salted key-derivation functions (Argon2id or PBKDF2 with minimum 600,000 iterations).

### Related Controls
- **AU-2**: Event Logging and Auditable Events
- **AU-6**: Audit Record Review, Analysis, and Reporting
- **SC-8**: Transmission Confidentiality and Integrity (TLS 1.3)
- **SC-13**: Cryptographic Protection (FIPS 140-3 validated modules)
