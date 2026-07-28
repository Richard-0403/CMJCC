/* Rater selection: POST the chosen name so the server can set the signed identity cookie.
 *
 * A fetch rather than a form post: python-multipart is not installed, so form parsing is not
 * available, and every write in this UI is a JSON endpoint for that reason.
 */
'use strict';

(function () {
  var statusRegion = document.getElementById('status');

  function announce(message) {
    if (statusRegion) {
      statusRegion.textContent = message;
    }
  }

  function post(url, body, describe) {
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(body || {})
    }).then(function (response) {
      return response.json().then(function (data) {
        return { ok: response.ok, status: response.status, data: data };
      }).catch(function () {
        return { ok: response.ok, status: response.status, data: {} };
      });
    }).then(function (result) {
      if (!result.ok) {
        announce('Could not ' + describe + ' (' + result.status + '): '
          + (result.data.detail || 'unexpected error'));
        return;
      }
      announce(describe + ' done. Loading your queue.');
      window.location.assign(result.data.next_url || '/');
    }).catch(function (error) {
      announce('Could not ' + describe + ': ' + error);
    });
  }

  Array.prototype.slice.call(document.querySelectorAll('.rater-button')).forEach(
    function (button) {
      button.addEventListener('click', function () {
        var raterId = button.getAttribute('data-rater-id');
        announce('Selecting ' + raterId + '...');
        post('/api/session', { rater_id: raterId }, 'select rater ' + raterId);
      });
    });

  var switchButton = document.getElementById('switch-rater');
  if (switchButton) {
    switchButton.addEventListener('click', function () {
      post('/api/session/clear', {}, 'switch rater');
    });
  }
}());
