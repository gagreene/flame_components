use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::Path;

use flame_components::Cell;
use serde::de::{self, Deserialize, Deserializer, MapAccess, Visitor};
use serde_json::Value;

pub(crate) struct Arguments(BTreeMap<String, Argument>);

impl Arguments {
    pub(crate) fn read(args: &[OsString]) -> Result<Self, String> {
        if args.iter().any(|arg| arg == "--json") {
            return match args {
                [flag, path] if flag == "--json" => Self::read_json(Path::new(path)),
                [flag] if flag == "--json" => Err("expected a path or - after --json".to_owned()),
                _ => Err("--json cannot be combined with other arguments".to_owned()),
            };
        }

        let mut arguments = Self(BTreeMap::new());
        let mut args = args.iter();
        while let Some(flag) = args.next() {
            let name = flag
                .to_str()
                .and_then(|flag| flag.strip_prefix("--"))
                .filter(|name| !name.is_empty())
                .ok_or_else(|| {
                    format!("expected an option of the form --name, received {flag:?}")
                })?;
            let argument = if name == "params-only" {
                Argument::Json(Value::Bool(true))
            } else {
                let value = args
                    .next()
                    .and_then(|arg| arg.to_str())
                    .filter(|arg| !arg.starts_with("--"))
                    .ok_or_else(|| format!("expected a value after --{name}"))?;
                Argument::Flag(value.to_owned())
            };
            arguments.insert(name.to_owned(), argument)?;
        }
        Ok(arguments)
    }

    fn read_json(path: &Path) -> Result<Self, String> {
        let data = if path == Path::new("-") {
            let mut data = Vec::new();
            io::stdin()
                .lock()
                .read_to_end(&mut data)
                .map_err(|error| format!("could not read JSON from stdin: {error}"))?;
            data
        } else {
            fs::read(path)
                .map_err(|error| format!("could not read JSON file {}: {error}", path.display()))?
        };
        serde_json::from_slice(&data).map_err(|error| format!("could not parse JSON: {error}"))
    }

    fn insert(&mut self, name: String, argument: Argument) -> Result<(), String> {
        use std::collections::btree_map::Entry;
        match self.0.entry(name) {
            Entry::Vacant(entry) => {
                entry.insert(argument);
                Ok(())
            }
            Entry::Occupied(entry) => Err(format!("--{} was supplied more than once", entry.key())),
        }
    }

    pub(crate) fn finish(&self) -> Result<(), String> {
        match self.0.keys().next() {
            Some(name) => Err(format!("unexpected argument --{name} for this calculation")),
            None => Ok(()),
        }
    }

    pub(crate) fn required(&mut self, name: &str) -> Result<Argument, String> {
        self.0
            .remove(name)
            .ok_or_else(|| format!("missing required argument --{name}"))
    }

    pub(crate) fn text(&mut self, name: &str) -> Result<String, String> {
        self.required(name)?.into_text(name)
    }

    pub(crate) fn optional_text(&mut self, name: &str) -> Result<Option<String>, String> {
        self.0
            .remove(name)
            .map(|arg| arg.into_text(name))
            .transpose()
    }

    pub(crate) fn scalar(&mut self, name: &str) -> Result<Scalar, String> {
        self.required(name)?.into_scalar(name)
    }

    pub(crate) fn cell(&mut self, name: &str) -> Result<Cell<f64>, String> {
        self.scalar(name).map(Scalar::into_cell)
    }

    pub(crate) fn optional_cell(&mut self, name: &str) -> Result<Option<Cell<f64>>, String> {
        self.0
            .remove(name)
            .map(|arg| arg.into_scalar(name).map(Scalar::into_cell))
            .transpose()
    }

    pub(crate) fn params_only(&mut self) -> Result<bool, String> {
        match self.0.remove("params-only") {
            None => Ok(false),
            Some(Argument::Json(Value::Bool(value))) => Ok(value),
            _ => Err("expected a boolean for params_only".to_owned()),
        }
    }
}

impl<'de> Deserialize<'de> for Arguments {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ArgumentsVisitor;

        impl<'de> Visitor<'de> for ArgumentsVisitor {
            type Value = Arguments;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object of scalar arguments")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut arguments = Arguments(BTreeMap::new());
                while let Some((name, value)) = map.next_entry::<String, Value>()? {
                    if value.is_array() || value.is_object() {
                        return Err(de::Error::custom(format!(
                            "expected a scalar JSON value for {name:?}"
                        )));
                    }
                    arguments
                        .insert(name.replace('_', "-"), Argument::Json(value))
                        .map_err(de::Error::custom)?;
                }
                Ok(arguments)
            }
        }

        deserializer.deserialize_map(ArgumentsVisitor)
    }
}

pub(crate) enum Argument {
    Flag(String),
    Json(Value),
}

impl Argument {
    pub(crate) fn as_text(&self) -> Option<&str> {
        match self {
            Self::Flag(value) | Self::Json(Value::String(value)) => Some(value),
            _ => None,
        }
    }

    fn into_text(self, name: &str) -> Result<String, String> {
        match self {
            Self::Flag(value) | Self::Json(Value::String(value)) => Ok(value),
            _ => Err(format!("expected a string for --{name}")),
        }
    }

    pub(crate) fn into_scalar(self, name: &str) -> Result<Scalar, String> {
        match self {
            Self::Flag(value) => match value.as_str() {
                "null" => Ok(Scalar::Missing),
                "true" => Ok(Scalar::Integer(1)),
                "false" => Ok(Scalar::Integer(0)),
                value => match Scalar::nonfinite(value) {
                    Some(value) => Ok(value),
                    None => Scalar::parse_number(value, name),
                },
            },
            Self::Json(Value::Null) => Ok(Scalar::Missing),
            Self::Json(Value::Bool(value)) => Ok(Scalar::Integer(i64::from(value))),
            // arbitrary_precision retains integer lexemes before any float conversion.
            Self::Json(Value::Number(value)) => Scalar::parse_number(&value.to_string(), name),
            Self::Json(Value::String(value)) => Scalar::nonfinite(&value).ok_or_else(|| {
                format!("expected a number, null, or non-finite token for --{name}")
            }),
            _ => Err(format!("expected a numeric scalar for --{name}")),
        }
    }
}

pub(crate) enum Scalar {
    Integer(i64),
    Float(f64),
    Missing,
}

impl Scalar {
    fn parse_number(value: &str, name: &str) -> Result<Self, String> {
        let digits = value
            .strip_prefix('-')
            .or_else(|| value.strip_prefix('+'))
            .unwrap_or(value);
        if !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return value.parse::<i64>().map(Self::Integer).map_err(|_| {
                format!(
                    "integer for --{name} is outside the signed 64-bit range; use a decimal or exponent for floating-point input"
                )
            });
        }
        let value = value
            .parse::<f64>()
            .map_err(|_| format!("expected a number, null, or non-finite token for --{name}"))?;
        if value.is_finite() {
            Ok(Self::Float(value))
        } else {
            Err(format!(
                "expected a finite floating-point number for --{name}; use Infinity, -Infinity, or NaN explicitly"
            ))
        }
    }

    fn nonfinite(value: &str) -> Option<Self> {
        match value {
            "Infinity" => Some(Self::Float(f64::INFINITY)),
            "-Infinity" => Some(Self::Float(f64::NEG_INFINITY)),
            "NaN" => Some(Self::Missing),
            _ => None,
        }
    }

    pub(crate) fn into_cell(self) -> Cell<f64> {
        match self {
            Self::Integer(value) => Cell::present(value as f64),
            Self::Float(value) => Cell::input(value),
            Self::Missing => Cell::missing(f64::NAN),
        }
    }
}
