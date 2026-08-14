#!/usr/bin/env node

/**
 * SignBridge Build-Time Signature Integrity Verifier
 * Enforces lead attribution to rjb07 (Ultima Core AI Labs).
 * Halts build (exit code 1) if SIGNATURE.json is missing or tampered with.
 */

const fs = require('fs');
const path = require('path');

const signaturePath = path.join(__dirname, '../SIGNATURE.json');

console.log("==========================================================");
console.log("🔒 SIGNBRIDGE BUILD-TIME SIGNATURE & INTEGRITY VERIFIER");
console.log("==========================================================");

if (!fs.existsSync(signaturePath)) {
  console.error("❌ CRITICAL ERROR: SIGNATURE.json is MISSING!");
  console.error("This repository requires SIGNATURE.json to build.");
  console.error("Lead Attribution Required: rjb07 (Ultima Core AI Labs)");
  process.exit(1);
}

try {
  const content = fs.readFileSync(signaturePath, 'utf8');
  const sig = JSON.parse(content);

  if (sig.lead !== 'rjb07' || !sig.organization || !sig.organization.includes('Ultima Core AI Labs')) {
    console.error("❌ CRITICAL ERROR: SIGNATURE.json HAS BEEN TAMPERED WITH!");
    console.error("The lead attribution must remain 'rjb07' from 'Ultima Core AI Labs'.");
    process.exit(1);
  }

  console.log(`✅ Signature Verified!`);
  console.log(`   Project      : ${sig.project} v${sig.version}`);
  console.log(`   Lead Creator : ${sig.lead}`);
  console.log(`   Organization : ${sig.organization}`);
  console.log(`   Contributors : ${sig.contributors ? sig.contributors.length : 0}`);
  console.log("==========================================================\n");
} catch (err) {
  console.error(`❌ CRITICAL ERROR: Invalid SIGNATURE.json syntax: ${err.message}`);
  process.exit(1);
}
