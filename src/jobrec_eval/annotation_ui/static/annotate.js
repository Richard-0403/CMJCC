/* Annotation screen behaviour: keyboard grading, and the client-side timing of each item.
 *
 * duration_ms is measured HERE and nowhere else. The clock starts when the item is rendered in
 * the browser and stops when the rater presses save, so it measures the time a person spent
 * looking at the item -- not server processing, not the gap while somebody fetched a coffee
 * between items. It is the only source for the annotation effort the thesis reports, so it is
 * posted with every single write.
 *
 * The keyboard map below is the whole interaction contract for a rater working at speed.
 * Shortcuts are suppressed while focus is in the notes box so typing a "0" into a note cannot
 * silently change the grade.
 */
'use strict';

var KEY_BINDINGS = {
  relevanceGrades: ['0', '1', '2', '3'],
  claimLabels: ['1', '0'],
  saveAndNext: ['Enter', 'n', 'N'],
  skip: ['s', 'S'],
  flagUnresolvableEvidence: ['e', 'E']
};

(function () {
  var root = document.getElementById('annotate');
  if (!root) {
    return;
  }

  /* performance.now() is monotonic, so a system clock change mid-item cannot produce a negative
     or wildly wrong duration. Date.now() is only the fallback. */
  var startedAt = (window.performance && window.performance.now)
    ? window.performance.now()
    : Date.now();

  var kind = root.getAttribute('data-kind') || '';
  var itemKey = root.getAttribute('data-item-key') || '';
  var saveUrl = root.getAttribute('data-save-url') || '/api/annotations';
  var skipUrl = root.getAttribute('data-skip-url') || '/queue';
  var queueUrl = root.getAttribute('data-queue-url') || '/queue';
  var flagValue = root.getAttribute('data-flag-value') || '';
  var allowedKeys = kind === 'claim' ? KEY_BINDINGS.claimLabels : KEY_BINDINGS.relevanceGrades;

  var statusRegion = document.getElementById('status');
  var readout = document.getElementById('selected-label');
  var notes = document.getElementById('notes');
  var flagBox = document.getElementById('flag-unresolvable');
  var saveButton = document.getElementById('save');
  var skipButton = document.getElementById('skip');
  var labelButtons = Array.prototype.slice.call(root.querySelectorAll('.label-button'));

  var selected = root.getAttribute('data-selected') || '';
  var saving = false;

  function announce(message) {
    if (statusRegion) {
      statusRegion.textContent = message;
    }
  }

  function elapsedMs() {
    var now = (window.performance && window.performance.now)
      ? window.performance.now()
      : Date.now();
    return Math.max(0, Math.round(now - startedAt));
  }

  function paintSelection() {
    labelButtons.forEach(function (button) {
      var isChosen = button.getAttribute('data-label') === selected;
      button.setAttribute('aria-pressed', isChosen ? 'true' : 'false');
    });
    if (readout) {
      readout.textContent = selected === '' ? 'none yet' : selected;
    }
  }

  function choose(value, announceChoice) {
    selected = String(value);
    paintSelection();
    if (announceChoice) {
      var chosen = labelButtons.filter(function (button) {
        return button.getAttribute('data-label') === selected;
      })[0];
      var description = chosen ? chosen.textContent.replace(/\s+/g, ' ').trim() : selected;
      announce('Selected ' + description + '. Press Enter to save and move on.');
    }
  }

  function currentFlags() {
    return (flagBox && flagBox.checked && flagValue) ? flagValue : '';
  }

  function save() {
    if (saving) {
      return;
    }
    if (selected === '') {
      announce(kind === 'claim'
        ? 'Choose 1 (supported) or 0 (not supported) before saving.'
        : 'Choose a grade from 0 to 3 before saving.');
      return;
    }
    saving = true;
    announce('Saving...');
    var body = {
      item_key: itemKey,
      label: parseInt(selected, 10),
      notes: notes ? notes.value : '',
      flags: currentFlags(),
      duration_ms: elapsedMs()
    };
    fetch(saveUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(body)
    }).then(function (response) {
      return response.json().then(function (data) {
        return { ok: response.ok, status: response.status, data: data };
      }).catch(function () {
        return { ok: response.ok, status: response.status, data: {} };
      });
    }).then(function (result) {
      saving = false;
      if (!result.ok) {
        announce('Not saved (' + result.status + '): '
          + (result.data.detail ? JSON.stringify(result.data.detail) : 'unexpected error')
          + '. Your answer is still on screen.');
        return;
      }
      announce('Saved label ' + result.data.label + ' after '
        + Math.round((result.data.duration_ms || 0) / 1000) + ' seconds. Loading the next item.');
      var next = result.data.next_url || queueUrl;
      window.setTimeout(function () {
        window.location.assign(next);
      }, 350);
    }).catch(function (error) {
      saving = false;
      announce('Not saved, the request failed: ' + error + '. Your answer is still on screen.');
    });
  }

  function skip() {
    announce('Skipped. This item stays in your queue and comes back later.');
    window.location.assign(skipUrl);
  }

  labelButtons.forEach(function (button) {
    button.addEventListener('click', function () {
      choose(button.getAttribute('data-label'), true);
    });
  });
  if (saveButton) {
    saveButton.addEventListener('click', save);
  }
  if (skipButton) {
    skipButton.addEventListener('click', skip);
  }
  if (flagBox) {
    flagBox.addEventListener('change', function () {
      announce(flagBox.checked
        ? 'Flag raised: a cited evidence id does not resolve.'
        : 'Flag cleared.');
    });
  }

  function typingInField(target) {
    if (!target || !target.tagName) {
      return false;
    }
    var tag = target.tagName.toLowerCase();
    return tag === 'textarea' || tag === 'input' || tag === 'select';
  }

  document.addEventListener('keydown', function (event) {
    if (event.ctrlKey || event.metaKey || event.altKey) {
      return;
    }
    if (typingInField(event.target)) {
      return;
    }
    var key = event.key;
    if (allowedKeys.indexOf(key) !== -1) {
      event.preventDefault();
      choose(key, true);
      return;
    }
    if (KEY_BINDINGS.saveAndNext.indexOf(key) !== -1) {
      event.preventDefault();
      save();
      return;
    }
    if (KEY_BINDINGS.skip.indexOf(key) !== -1) {
      event.preventDefault();
      skip();
      return;
    }
    if (flagBox && KEY_BINDINGS.flagUnresolvableEvidence.indexOf(key) !== -1) {
      event.preventDefault();
      flagBox.checked = !flagBox.checked;
      announce(flagBox.checked
        ? 'Flag raised: a cited evidence id does not resolve.'
        : 'Flag cleared.');
    }
  });

  paintSelection();
  if (root.getAttribute('data-was-saved') === 'true') {
    /* Setting the text after load is a CHANGE to the aria-live region, which is what makes a
       screen reader announce it; static initial content would not be announced. */
    announce('Previous item saved. This is the next item in your queue.');
  } else if (selected !== '') {
    announce('You already answered this item with ' + selected + '. Saving again revises it.');
  }
}());
