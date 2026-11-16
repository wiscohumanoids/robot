// Concatenates src/* into two outputs:
//   index.html        standalone page (double-click to open, or host anywhere)
//   artifact.html     fragment for the Artifact publisher (no doctype/html/head/body)
const fs = require('fs'), path = require('path');
const src = f => fs.readFileSync(path.join(__dirname, 'src', f), 'utf8');
const js = ['core.js', 'pages.js', 'ch_map.js', 'ch_walk1.js', 'ch_walk2.js', 'ch_arms.js', 'ch_see.js', 'ch_ctl.js', 'ch_body.js']
  .map(src).join('\n').replace(/^'use strict';\n/gm, '') + '\nboot();\n';
if (/<\/script/i.test(js)) throw new Error('JS contains </script');
const css = src('style.css');
const fonts = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,500..900&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">';
const head = `<title>WisCortex</title>\n${fonts}\n<style>\n${css}\n</style>`;
const body = `<div id="root"></div>\n<script>\n'use strict';\n${js}</script>`;
fs.writeFileSync(path.join(__dirname, 'artifact.html'), head + '\n' + body + '\n');
fs.writeFileSync(path.join(__dirname, 'index.html'), `<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n${head}\n</head>\n<body>\n${body}\n</body>\n</html>\n`);
console.log('built', (fs.statSync(path.join(__dirname, 'index.html')).size / 1024).toFixed(0) + ' KB');
