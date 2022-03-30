let socket = new WebSocket("ws://localhost:8000/events");

socket.onmessage = function(event) {
  const currentSlide = Reveal.getCurrentSlide();
  switch(event.data){
    case "right":
      console.log("received 'right' event");
      Reveal.right();
      break;
    case "left":
      console.log("received 'left' event");
      Reveal.left();
      break;
    case "up":
      console.log("received 'up' event");
      Reveal.right();
      break;
    case "down":
      console.log("received 'down' event");
      Reveal.left();
      break;
    case "rotate":
      console.log("received 'rotate' event");
      rotateRotatables(currentSlide, 90);  // defined in helper_methods.js
      break;
    case "rotate_left":
      console.log("received 'rotate_left' event");
      rotateRotatables(currentSlide, -90);  // defined in helper_methods.js
      break;
    case "rotate180":
      console.log("received 'rotate180' event");
      rotateRotatables(currentSlide, 180);  // defined in helper_methods.js
      break;
    case "rotate360":
      console.log("received 'rotate360' event");
      rotateRotatables(currentSlide, 360);  // defined in helper_methods.js
      break;
    case "zoom_in":
      console.log("received 'zoom_in' event");

      // increases zoom by 10%
      zoom(10); // `zoom()` is defined in helper_methods.js
      break;
    case "zoom_out":
      console.log("received 'zoom_out' event");

      // decreases zoom by 10%
      zoom(-10); // `zoom()` is defined in helper_methods.js
      break;
    default:
      console.debug(`unknown message received from server: ${event.data}`);
  }
};
