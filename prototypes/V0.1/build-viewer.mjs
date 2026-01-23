/**
 * Build script to extract SuperSplat viewer files from npm package
 * and prepare them for Flask serving (ES Module version)
 */

import { html, css, js } from '@playcanvas/supersplat-viewer';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const staticDir = join(__dirname, 'static', 'viewer');

// Create directories
if (!existsSync(staticDir)) {
    mkdirSync(staticDir, { recursive: true });
}

// Write the files
writeFileSync(join(staticDir, 'index.html'), html);
writeFileSync(join(staticDir, 'index.css'), css);
writeFileSync(join(staticDir, 'index.js'), js);

console.log('SuperSplat viewer files extracted to static/viewer/');
console.log('  - index.html');
console.log('  - index.css');
console.log('  - index.js');
