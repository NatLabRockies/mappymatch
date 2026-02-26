# Mappymatch

Mappymatch is a pure-Python package developed and open-sourced by the National Laboratory of the Rockies. It contains a collection of "Matchers" that enable matching a GPS trace (series of GPS coordinates) to a map.

## The Current Matchers

- **`LCSSMatcher`**: A matcher that implements the LCSS algorithm described in this [paper](https://doi.org/10.3141%2F2645-08). Works best with high-resolution GPS traces.
- **`OsrmMatcher`**: A light matcher that pings an OSRM server to request map matching results. See the [official documentation](http://project-osrm.org/) for more info.
- **`ValhallaMatcher`**: A matcher to ping a [Valhalla](https://www.interline.io/valhalla/) server for map matching results.

## Currently Supported Map Formats

- **Open Street Maps**
