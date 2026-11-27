#![forbid(unsafe_code)]

mod arguments;

use std::env;
use std::io::{self, Write};
use std::process::ExitCode;

use flame_components::{
    depth::flame_depth,
    height::{flame_height, HeightModel},
    length::{flame_length, LengthModel},
    residence::{flame_residence_time, ResidenceUnits},
    tilt::{flame_tilt, TiltModel},
    wind::{mid_flame_ws, WindUnits},
    Cell, FireType, SlopeUnits, WindSpeedUnits,
};
use serde_json::{json, Number, Value};

use arguments::{Arguments, Scalar};

const HELP: &str = "Usage: flame-components <command> [--flag value ... | --json PATH]

Commands:
  mid-flame-ws          Calculate mid-flame wind speed
  flame-length          Calculate flame length or list model coefficients
  flame-height          Calculate flame height
  flame-tilt            Calculate flame tilt
  flame-residence-time  Calculate flame residence time
  flame-depth           Calculate flame depth

Use --json - to read a scalar JSON object from stdin.
Use flame-components <command> --help for required flags.
";

const WIND_HELP: &str = "\
mid-flame-ws --wind-speed <number> --canopy-cover <percent> --canopy-ht <number>
             --canopy-baseht <number> [--units <SI|IMP>]
Result in m/s. SI (default) takes km/h and metres. IMP takes mi/h and feet.
";

const LENGTH_HELP: &str = "\
flame-length --model <selector> --fire-intensity <kW/m> [--flame-depth <m>]
flame-length --model <selector> --params-only
Result in metres or coefficients in model order. Finney_HEAD requires flame depth.
";

const HEIGHT_HELP: &str = "\
flame-height --model Nelson --flame-length <m> --fire-type <1|2|3>
             --fire-intensity <kW/m> --midflame-ws <m/s>
flame-height --model Finney --flame-length <m> --flame-tilt <degrees>
             --slope-angle <number> --slope-units <degrees|percent>
Result in metres.
";

const TILT_HELP: &str = "\
flame-tilt --model <Standard|Finney> --flame-length <m> --flame-height <m>
           [--slope-angle <number> --slope-units <degrees|percent>]
flame-tilt --model Butler --wind-speed <number> --wind-speed-units <kph|mps|mph>
           --canopy-ht <m>
Result in degrees. Slope flags are required for Finney.
";

const RESIDENCE_HELP: &str = "\
flame-residence-time --ros <m/min> --fuel-consumption <kg/m²> --midflame-ws <m/s>
                     --units <sec|min>
Result in the requested units.
";

const DEPTH_HELP: &str = "\
flame-depth --ros <m/min> --res-time <min>
Result in metres.
";

fn main() -> ExitCode {
    match run() {
        Ok(output) => {
            if let Err(error) = writeln!(io::stdout().lock(), "{output}") {
                eprintln!("error: could not write output: {error}");
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<String, String> {
    let mut arguments = env::args_os().skip(1);
    let command = arguments
        .next()
        .ok_or("expected a calculation subcommand; use --help")?;
    let args: Vec<_> = arguments.collect();
    let command = command
        .to_str()
        .ok_or("the calculation subcommand must be Unicode")?;
    if args.is_empty() {
        match command {
            "--help" | "-h" => return Ok(HELP.trim_end().to_owned()),
            "--version" | "-V" => {
                return Ok(format!("flame-components {}", env!("CARGO_PKG_VERSION")))
            }
            _ => {}
        }
    }
    let help = match command {
        "mid-flame-ws" => WIND_HELP,
        "flame-length" => LENGTH_HELP,
        "flame-height" => HEIGHT_HELP,
        "flame-tilt" => TILT_HELP,
        "flame-residence-time" => RESIDENCE_HELP,
        "flame-depth" => DEPTH_HELP,
        _ => return Err(format!("unknown subcommand {command:?}; use --help")),
    };
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        return Ok(help.trim_end().to_owned());
    }

    let mut args = Arguments::read(&args)?;
    let mut substituted_canopy_height = false;
    let value = match command {
        "mid-flame-ws" => {
            let units = match args.optional_text("units")?.as_deref().unwrap_or("SI") {
                "SI" => WindUnits::Si,
                "IMP" => WindUnits::Imperial,
                _ => return Err("expected --units SI or IMP".to_owned()),
            };
            let result = mid_flame_ws(
                args.cell("wind-speed")?,
                args.cell("canopy-cover")?,
                args.cell("canopy-ht")?,
                args.cell("canopy-baseht")?,
                units,
            );
            substituted_canopy_height = result.substituted_canopy_height;
            float_result(result.value.nonnegative())
        }
        "flame-length" => {
            let model = args
                .text("model")?
                .parse::<LengthModel>()
                .map_err(|error| error.to_string())?;
            let depth = args.optional_cell("flame-depth")?;
            if args.params_only()? {
                args.optional_cell("fire-intensity")?;
                let (coefficient, intensity, depth) = model.coefficients();
                match depth {
                    Some(depth) => json!([coefficient, intensity, depth]),
                    None => json!([coefficient, intensity]),
                }
            } else {
                let result = flame_length(model, args.cell("fire-intensity")?, depth)
                    .map_err(|error| error.to_string())?;
                float_result(result.nonnegative())
            }
        }
        "flame-height" => {
            let length = args.cell("flame-length")?;
            let model = match args.text("model")?.as_str() {
                "Nelson" => HeightModel::Nelson {
                    fire_type: args.fire_type()?,
                    fire_intensity: args.cell("fire-intensity")?,
                    midflame_ws: args.cell("midflame-ws")?,
                },
                "Finney" => HeightModel::Finney {
                    flame_tilt: args.cell("flame-tilt")?,
                    slope_angle: args.cell("slope-angle")?,
                    slope_units: args.slope_units()?,
                },
                _ => return Err("expected --model Nelson or Finney".to_owned()),
            };
            let result = flame_height(length, model).map_err(|error| error.to_string())?;
            float_result(result.nonnegative())
        }
        "flame-tilt" => {
            let model = match args.text("model")?.as_str() {
                "Standard" => TiltModel::Standard {
                    flame_length: args.cell("flame-length")?,
                    flame_height: args.cell("flame-height")?,
                },
                "Finney" => TiltModel::Finney {
                    flame_length: args.cell("flame-length")?,
                    flame_height: args.cell("flame-height")?,
                    slope_angle: args.cell("slope-angle")?,
                    slope_units: args.slope_units()?,
                },
                "Butler" => TiltModel::Butler {
                    wind_speed: args.cell("wind-speed")?,
                    wind_speed_units: args.wind_units()?,
                    canopy_ht: args.cell("canopy-ht")?,
                },
                _ => return Err("expected --model Standard, Finney, or Butler".to_owned()),
            };
            let result = flame_tilt(model).map_err(|error| error.to_string())?;
            float_result(result.nonnegative())
        }
        "flame-residence-time" => {
            let units = match args.text("units")?.as_str() {
                "sec" => ResidenceUnits::Seconds,
                "min" => ResidenceUnits::Minutes,
                _ => return Err("expected --units sec or min".to_owned()),
            };
            let result = flame_residence_time(
                args.cell("ros")?,
                args.cell("fuel-consumption")?,
                args.cell("midflame-ws")?,
                units,
            );
            float_result(result.nonnegative())
        }
        "flame-depth" => {
            let ros = args.scalar("ros")?;
            let res_time = args.scalar("res-time")?;
            match (ros, res_time) {
                (Scalar::Integer(ros), Scalar::Integer(res_time)) => {
                    json!(
                        flame_depth(Cell::present(ros), Cell::present(res_time))
                            .nonnegative()
                            .value
                    )
                }
                (ros, res_time) => {
                    float_result(flame_depth(ros.into_cell(), res_time.into_cell()).nonnegative())
                }
            }
        }
        _ => unreachable!("validated subcommand"),
    };
    args.finish()?;
    if substituted_canopy_height {
        eprintln!("warning: zero canopy height was replaced with 1.6404 ft (0.5 m) in the Albini & Baughman (1979) equation");
    }
    Ok(value.to_string())
}

fn float_result(cell: Cell<f64>) -> Value {
    if cell.masked {
        Value::Null
    } else if cell.value.is_nan() {
        json!("NaN")
    } else if cell.value == f64::INFINITY {
        json!("Infinity")
    } else if cell.value == f64::NEG_INFINITY {
        json!("-Infinity")
    } else {
        Value::Number(Number::from_f64(cell.value).expect("finite value"))
    }
}

impl Arguments {
    fn fire_type(&mut self) -> Result<Cell<f64>, String> {
        let argument = self.required("fire-type")?;
        Ok(match argument.as_text() {
            Some("surface") => FireType::Surface.into(),
            Some("passive crown") => FireType::PassiveCrown.into(),
            Some("active crown") => FireType::ActiveCrown.into(),
            _ => argument.into_scalar("fire-type")?.into_cell(),
        })
    }

    fn slope_units(&mut self) -> Result<SlopeUnits, String> {
        match self.text("slope-units")?.as_str() {
            "degrees" => Ok(SlopeUnits::Degrees),
            "percent" => Ok(SlopeUnits::Percent),
            _ => Err("expected --slope-units degrees or percent".to_owned()),
        }
    }

    fn wind_units(&mut self) -> Result<WindSpeedUnits, String> {
        match self.text("wind-speed-units")?.as_str() {
            "kph" => Ok(WindSpeedUnits::Kph),
            "mps" => Ok(WindSpeedUnits::Mps),
            "mph" => Ok(WindSpeedUnits::Mph),
            _ => Err("expected --wind-speed-units kph, mps, or mph".to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::float_result;
    use flame_components::Cell;
    use serde_json::{json, Value};

    #[test]
    fn encodes_nonfinite_payloads_independently_of_the_mask() {
        for (nonfinite, expected) in [
            (f64::NAN, "NaN"),
            (f64::INFINITY, "Infinity"),
            (f64::NEG_INFINITY, "-Infinity"),
        ] {
            assert_eq!(float_result(Cell::present(nonfinite)), json!(expected));
            assert_eq!(float_result(Cell::missing(nonfinite)), Value::Null);
        }
    }
}
