/* Adjudication queue: record a final label with its required written reason.
 *
 * The reason is checked here for a quick message and again on the server, which is the check that
 * counts: an adjudicated label overrides two human judgements and is reported, so it does not get
 * recorded without a justification.
 */
'use strict';

(function () {
  var root = document.getElementById('adjudication');
  if (!root) {
    return;
  }
  var verdictUrl = root.getAttribute('data-verdict-url') || '/api/adjudication/verdicts';
  var sessionUrl = root.getAttribute('data-session-url') || '/api/adjudication/session';
  var nameField = document.getElementById('adjudicator');
  var statusRegion = document.getElementById('status');

  function announce(message) {
    if (statusRegion) {
      statusRegion.textContent = message;
    }
  }

  function rememberName() {
    var name = nameField ? nameField.value.trim() : '';
    if (!name) {
      return;
    }
    fetch(sessionUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ adjudicator: name })
    }).catch(function () { /* attribution convenience only, never fatal */ });
  }

  if (nameField) {
    nameField.addEventListener('change', rememberName);
  }

  Array.prototype.slice.call(root.querySelectorAll('.adjudication-case')).forEach(
    function (caseNode) {
      var itemKey = caseNode.getAttribute('data-item-key');
      var reason = caseNode.querySelector('.reason');
      var caseStatus = caseNode.querySelector('.case-status');

      function report(message) {
        if (caseStatus) {
          caseStatus.textContent = message;
        }
        announce(message);
      }

      Array.prototype.slice.call(caseNode.querySelectorAll('.verdict-button')).forEach(
        function (button) {
          button.addEventListener('click', function () {
            var name = nameField ? nameField.value.trim() : '';
            if (!name) {
              report('Enter an adjudicator name first: the verdict is recorded against it.');
              if (nameField) {
                nameField.focus();
              }
              return;
            }
            var text = reason ? reason.value.trim() : '';
            if (!text) {
              report('A written reason is required before a final label can be recorded.');
              if (reason) {
                reason.focus();
              }
              return;
            }
            var label = button.getAttribute('data-label');
            report('Recording final label ' + label + '...');
            fetch(verdictUrl, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              credentials: 'same-origin',
              body: JSON.stringify({
                item_key: itemKey,
                final_label: parseInt(label, 10),
                reason: text,
                adjudicator: name
              })
            }).then(function (response) {
              return response.json().then(function (data) {
                return { ok: response.ok, status: response.status, data: data };
              }).catch(function () {
                return { ok: response.ok, status: response.status, data: {} };
              });
            }).then(function (result) {
              if (!result.ok) {
                report('Not recorded (' + result.status + '): '
                  + (result.data.detail ? JSON.stringify(result.data.detail) : 'unexpected error'));
                return;
              }
              report('Recorded final label ' + result.data.final_label + ' by '
                + result.data.adjudicator + '. ' + result.data.open_cases
                + ' case(s) still open.');
            }).catch(function (error) {
              report('Not recorded, the request failed: ' + error);
            });
          });
        });
    });
}());
