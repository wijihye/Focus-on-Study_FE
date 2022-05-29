window.addEventListener('load', function () {
  showClock();
});

let Target = document.querySelector('#clock');
let End = document.querySelector('#end_btn');

let hours = 0,
  minutes = 0,
  seconds = 0;

function showClock() {
  seconds += 1;
  if (seconds == 60) {
    (minutes += 1), (seconds = 0);
  }
  if (minutes == 60) {
    (hours += 1), (minutes = 0);
  }

  let msg = hours + '시 ';
  msg += minutes + '분 ';
  msg += seconds + '초';

  console.log(seconds);
  Target.innerText = msg;
}

var timer = setInterval(showClock, 1000);

End.addEventListener('click', () => {
  clearInterval(timer);
  console.log(hours + ':' + minutes + ':' + seconds);
});
