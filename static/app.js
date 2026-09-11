/* Image2Genre — UI behaviour.
   Four small jobs: the dropzone, the analyzing view, the image/playlist tabs,
   and mounting a YouTube iframe only when a card is actually clicked. */
(function () {
  'use strict';

  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  /* ---------------------------- dropzone ---------------------------- */
  var form = $('#upload-form');
  if (form) {
    var input = $('#image-input');
    var zone = $('#dropzone');
    var submit = $('#submit-btn');
    var title = zone && $('.drop__title', zone);
    var originalTitle = title ? title.textContent : '';

    var accept = function (files) {
      if (!files || !files.length) return;
      input.files = files;
      zone.classList.add('has-file');
      if (title) title.textContent = files[0].name;
      if (submit) submit.hidden = false;
    };

    input.addEventListener('change', function () {
      if (input.files && input.files.length) accept(input.files);
      else {
        zone.classList.remove('has-file');
        if (title) title.textContent = originalTitle;
        if (submit) submit.hidden = true;
      }
    });

    ['dragenter', 'dragover'].forEach(function (evt) {
      zone.addEventListener(evt, function (e) { e.preventDefault(); zone.classList.add('is-over'); });
    });
    ['dragleave', 'drop'].forEach(function (evt) {
      zone.addEventListener(evt, function (e) { e.preventDefault(); zone.classList.remove('is-over'); });
    });
    zone.addEventListener('drop', function (e) {
      if (e.dataTransfer && e.dataTransfer.files) accept(e.dataTransfer.files);
    });

    /* ------------------------- analyzing view -------------------------
       The server returns finished HTML in one response, so these step
       timings are estimates of a typical run, not real progress. They exist
       to fill the 5-10s round trip instead of showing a blank page. */
    form.addEventListener('submit', function () {
      var overlay = $('#analyzing');
      if (!overlay || !input.files || !input.files.length) return;

      var preview = $('#analyzing-image');
      if (preview) {
        var url = URL.createObjectURL(input.files[0]);
        preview.src = url;
        preview.addEventListener('load', function () { URL.revokeObjectURL(url); }, { once: true });
      }

      overlay.hidden = false;
      document.body.style.overflow = 'hidden';

      var steps = $$('.step', overlay);
      var fill = $('#progress-fill');
      // [delay before this step becomes active, progress % once it is]
      var schedule = [[0, 12], [500, 34], [3200, 62], [5200, 88]];

      schedule.forEach(function (entry, i) {
        setTimeout(function () {
          steps.forEach(function (s, j) {
            s.classList.toggle('is-done', j < i);
            s.classList.toggle('is-active', j === i);
          });
          if (fill) fill.style.width = entry[1] + '%';
        }, entry[0]);
      });
    });
  }

  /* ------------------- source / attention image tabs ------------------- */
  var frames = $('.frames');
  if (frames) {
    $$('.frames__tab', frames).forEach(function (btn) {
      btn.addEventListener('click', function () {
        frames.dataset.frame = btn.dataset.show;
        $$('.frames__tab', frames).forEach(function (b) {
          var on = b === btn;
          b.classList.toggle('is-active', on);
          b.setAttribute('aria-selected', String(on));
        });
      });
    });
  }

  /* --------------------- global / khaleeji + arrows --------------------- */
  var rail = $('.rail');
  if (rail) {
    $$('.seg__btn', rail).forEach(function (btn) {
      btn.addEventListener('click', function () {
        rail.dataset.tab = btn.dataset.tab;
        $$('.seg__btn', rail).forEach(function (b) {
          var on = b === btn;
          b.classList.toggle('is-active', on);
          b.setAttribute('aria-selected', String(on));
        });
      });
    });

    $$('.arrow', rail).forEach(function (btn) {
      btn.addEventListener('click', function () {
        var strip = $('.rail__strip[data-group="' + rail.dataset.tab + '"]', rail);
        if (!strip) return;
        // In RTL, scrolling "forward" means moving towards negative scrollLeft.
        var rtl = getComputedStyle(strip).direction === 'rtl' ? -1 : 1;
        var step = Math.max(strip.clientWidth * 0.8, 204);
        strip.scrollBy({ left: rtl * (btn.dataset.scroll === 'forward' ? step : -step), behavior: 'smooth' });
      });
    });
  }

  /* ----------- mount a YouTube iframe only on an actual click ----------- */
  document.addEventListener('click', function (e) {
    var poster = e.target.closest && e.target.closest('.card__poster');
    if (!poster) return;
    var id = poster.dataset.video;
    if (!id) return;

    var frame = document.createElement('iframe');
    frame.className = 'card__frame';
    frame.src = 'https://www.youtube.com/embed/' + encodeURIComponent(id) + '?autoplay=1&rel=0';
    frame.title = poster.getAttribute('aria-label') || '';
    frame.allow = 'accelerometer; autoplay; encrypted-media; picture-in-picture';
    frame.allowFullscreen = true;
    frame.referrerPolicy = 'strict-origin-when-cross-origin';
    poster.replaceWith(frame);
  });
})();
