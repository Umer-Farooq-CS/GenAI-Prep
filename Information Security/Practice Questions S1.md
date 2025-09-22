# Information Security Practice Questions – S1

---

## 📝 MULTIPLE CHOICE QUESTIONS

1. Which principle of the CIA triad ensures data is accurate and unaltered?
   a) Confidentiality
   b) Availability
   c) Non-repudiation
   d) Integrity
   
   Answer: d) Integrity ✓

2. Which of the following best protects data in transit over an untrusted network?
   a) AES at rest
   b) RAID 5
   c) TLS/SSL
   d) Access Control Lists
   
   Answer: c) TLS/SSL ✓

3. In RBAC, access is primarily determined by:
   a) User identity only
   b) Ownership of the object
   c) Assigned role(s)
   d) Mandatory labels
   
   Answer: c) Assigned role(s) ✓

4. A salted password hash primarily defends against:
   a) SQL injection
   b) Rainbow table attacks
   c) Buffer overflows
   d) Cross-site scripting
   
   Answer: b) Rainbow table attacks ✓

5. Which attack floods a system with requests to make it unavailable?
   a) MITM
   b) DoS/DDoS
   c) Phishing
   d) Privilege escalation
   
   Answer: b) DoS/DDoS ✓

6. What is the primary purpose of a digital signature?
   a) Confidentiality only
   b) Integrity and authentication (and non‑repudiation)
   c) Availability
   d) Key exchange
   
   Answer: b) Integrity and authentication (and non‑repudiation) ✓

7. Which encryption mode provides both confidentiality and integrity if used with an auth tag?
   a) ECB
   b) CBC without MAC
   c) GCM
   d) CTR without MAC
   
   Answer: c) GCM ✓

8. A security policy enforcing fixed labels like Top Secret/Secret is an example of:
   a) DAC
   b) RBAC
   c) MAC
   d) ABAC
   
   Answer: c) MAC ✓

9. Which best practice mitigates SQL injection?
   a) Client-side validation only
   b) Parameterized queries/prepared statements
   c) String concatenation with sanitization
   d) Output encoding
   
   Answer: b) Parameterized queries/prepared statements ✓

10. Which control is detective rather than preventive?
    a) Firewall rule
    b) IDS alert
    c) MFA
    d) Input validation
    
    Answer: b) IDS alert ✓

---

## 📋 SHORT QUESTIONS

1. Explain the differences among Authentication, Authorization, and Accounting (AAA).
   
   Answer: Authentication verifies identity; Authorization grants permitted actions; Accounting logs and audits activity for traceability.

2. What are the three states of data and one control for each?
   
   Answer: At rest → disk encryption; In transit → TLS/VPN; In use → secure enclaves/access controls.

3. How do hashing and encryption differ?
   
   Answer: Hashing is one‑way for integrity (e.g., SHA‑256); encryption is reversible with keys for confidentiality (e.g., AES/RSA).

4. Give one countermeasure each for phishing, DDoS, and ransomware.
   
   Answer: Phishing → user training + email filters; DDoS → rate limiting/CDN; Ransomware → backups + EDR.

5. Why are IVs/nonces critical in symmetric encryption modes like CTR/GCM?
   
   Answer: Reusing nonces leaks keystream, enabling plaintext recovery or forgeries; uniqueness prevents reuse attacks.

6. What is network segmentation and why is it useful?
   
   Answer: Dividing networks into zones/VLANs to limit lateral movement and blast radius of breaches.

7. Define least privilege and give a practical example.
   
   Answer: Grant only required permissions; e.g., a service account with read‑only access to one S3 bucket.

8. How do HMACs provide integrity and authenticity?
   
   Answer: HMAC uses a shared secret and hash; only parties with the key can create a valid tag.

9. What is the principle behind zero trust?
   
   Answer: Never trust, always verify; continuous authentication/authorization regardless of network location.

10. Differentiate IDS vs IPS.
    
    Answer: IDS detects and alerts; IPS sits inline to detect and actively block.

---

## 🧮 LONG & NUMERICAL QUESTIONS

1) Subnetting and Addressing

Problem: Given the network 192.168.10.0/24, you need at least 5 subnets with roughly equal host capacity. Provide the new subnet mask, list the first two subnets with usable host ranges, and calculate usable hosts per subnet.

Solution:
- Need 5 subnets → borrow 3 bits (2^3 = 8 ≥ 5)
- New mask: /27 (255.255.255.224)
- Hosts per /27: 2^(32−27) − 2 = 32 − 2 = 30
- Subnet increments: 32
- Subnet 1: 192.168.10.0/27 → usable 192.168.10.1 – 192.168.10.30
- Subnet 2: 192.168.10.32/27 → usable 192.168.10.33 – 192.168.10.62

2) Access Control Matrix Calculation

Problem: Users U1, U2; Objects F1, F2. Initial rights: U1:{F1:R,W}, U2:{F1:R}, U1:{F2:-}, U2:{F2:W}. Apply: (a) Revoke U1 write on F1; (b) Grant U1 read on F2; (c) Remove U2 write on F2. Provide final matrix.

Solution:
- Start:
  - F1: U1 = R,W; U2 = R
  - F2: U1 = –; U2 = W
- (a) U1 on F1 → remove W: U1 = R; U2 = R
- (b) U1 on F2 → add R: U1 = R; U2 = W
- (c) U2 on F2 → remove W: U1 = R; U2 = –
- Final:
  - F1: U1 = R; U2 = R
  - F2: U1 = R; U2 = –

3) Cryptography – AES‑GCM Tag Verification

Problem: Given ciphertext C, nonce N, key K, and tag T, describe verification steps and identify what fails if the nonce repeats with same key.

Solution:
1. Recompute tag = GCM(K, N, AAD, C)
2. Compare constant‑time with received T; reject if mismatch
3. If nonce repeats with same K, keystream reuse allows attackers to XOR ciphertexts to recover XOR of plaintexts; integrity may also fail due to tag forgeries.

4) Risk Assessment Calculation

Problem: Asset value = $100,000; Single Loss Expectancy (SLE) for a threat is 15%. Annual Rate of Occurrence (ARO) = 0.2. Compute Annualized Loss Expectancy (ALE) and expected loss over 3 years.

Solution:
- SLE = Asset × Exposure Factor = 100,000 × 0.15 = $15,000
- ALE = SLE × ARO = 15,000 × 0.2 = $3,000 per year
- 3‑year expected loss (assuming independence, linear expectation) = 3 × 3,000 = $9,000

5) Hashing – Collision Probability (Birthday Approximation)

Problem: For a 128‑bit hash, estimate number of samples n for ~50% collision probability.

Solution:
- Birthday bound: n ≈ 1.177 × √(2^128) = 1.177 × 2^64 ≈ 2.17 × 10^19 samples

---

This S1 set mirrors the structure of your Generative AI practice file: MCQs, short answers, and numericals, each with solutions for quick revision.


