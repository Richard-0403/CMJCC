/* Export screen: run the export and show the paths, row counts, skipped items and hashes. */
'use strict';

(function () {
  var root = document.getElementById('export');
  if (!root) {
    return;
  }
  var exportUrl = root.getAttribute('data-export-url') || '/api/export';
  var button = document.getElementById('run-export');
  var resultNode = document.getElementById('export-result');
  var statusRegion = document.getElementById('status');

  function announce(message) {
    if (statusRegion) {
      statusRegion.textContent = message;
    }
  }

  function row(label, value) {
    var tr = document.createElement('tr');
    var th = document.createElement('th');
    th.setAttribute('scope', 'row');
    th.textContent = label;
    var td = document.createElement('td');
    td.textContent = value;
    tr.appendChild(th);
    tr.appendChild(td);
    return tr;
  }

  function render(data) {
    var table = document.createElement('table');
    table.className = 'fields';
    var caption = document.createElement('caption');
    caption.textContent = 'Export ' + data.export_id;
    table.appendChild(caption);
    var body = document.createElement('tbody');
    body.appendChild(row('Relevance CSV', data.relevance_csv));
    body.appendChild(row('Claims CSV', data.claims_csv));
    body.appendChild(row('Archive dump', data.dump));
    body.appendChild(row('Manifest', data.manifest));
    Object.keys(data.row_counts || {}).forEach(function (name) {
      body.appendChild(row('Rows in ' + name, data.row_counts[name]));
    });
    Object.keys(data.incomplete || {}).forEach(function (kind) {
      body.appendChild(row('Incomplete ' + kind + ' items skipped', data.incomplete[kind]));
    });
    Object.keys(data.sha256 || {}).forEach(function (name) {
      body.appendChild(row('SHA-256 of ' + name, data.sha256[name]));
    });
    var counts = data.counts || {};
    Object.keys(counts).forEach(function (name) {
      body.appendChild(row(name.replace(/_/g, ' '), counts[name]));
    });
    table.appendChild(body);
    resultNode.textContent = '';
    resultNode.appendChild(table);
  }

  if (button) {
    button.addEventListener('click', function () {
      announce('Running the export...');
      button.disabled = true;
      fetch(exportUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: '{}'
      }).then(function (response) {
        return response.json().then(function (data) {
          return { ok: response.ok, status: response.status, data: data };
        }).catch(function () {
          return { ok: response.ok, status: response.status, data: {} };
        });
      }).then(function (result) {
        button.disabled = false;
        if (!result.ok) {
          announce('Export failed (' + result.status + '): '
            + (result.data.detail ? JSON.stringify(result.data.detail) : 'unexpected error'));
          return;
        }
        render(result.data);
        announce('Export written. Paths and hashes are listed under Result.');
      }).catch(function (error) {
        button.disabled = false;
        announce('Export failed: ' + error);
      });
    });
  }
}());
