"""Generate 500 assumption questions proportional to domain distribution,
then fetch values via Tavily and save to datasets/fermi_facts.json.

Run:
    source .env && python3 scripts/generate_assumption_questions.py --dry-run
    source .env && python3 scripts/generate_assumption_questions.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "datasets" / "fermi_facts.json"

# ---------------------------------------------------------------------------
# 500 assumption questions, proportional to domain distribution:
#   biology_medicine    ~128  (25.5%)
#   physics_engineering ~111  (22.2%)
#   other               ~ 94  (18.8%)
#   earth_science       ~ 63  (12.5%)
#   astronomy           ~ 38  ( 7.6%)
#   chemistry           ~ 36  ( 7.3%)
#   mathematics         ~ 30  ( 6.0%)
# ---------------------------------------------------------------------------

ASSUMPTION_QUESTIONS: list[dict] = [

    # -----------------------------------------------------------------------
    # PHYSICS & ENGINEERING (111)
    # -----------------------------------------------------------------------
    {"domain": "physics_engineering", "q": "What is the speed of light in meters per second?"},
    {"domain": "physics_engineering", "q": "What is the mass of a proton in kilograms?"},
    {"domain": "physics_engineering", "q": "What is the mass of an electron in kilograms?"},
    {"domain": "physics_engineering", "q": "What is the mass of a neutron in kilograms?"},
    {"domain": "physics_engineering", "q": "What is the gravitational constant G in SI units?"},
    {"domain": "physics_engineering", "q": "What is the Planck constant in joule-seconds?"},
    {"domain": "physics_engineering", "q": "What is the Boltzmann constant in joules per kelvin?"},
    {"domain": "physics_engineering", "q": "What is the elementary charge in coulombs?"},
    {"domain": "physics_engineering", "q": "What is the permittivity of free space in SI units?"},
    {"domain": "physics_engineering", "q": "What is the permeability of free space in SI units?"},
    {"domain": "physics_engineering", "q": "What is the Stefan-Boltzmann constant in SI units?"},
    {"domain": "physics_engineering", "q": "What is the Avogadro constant?"},
    {"domain": "physics_engineering", "q": "What is the universal gas constant in joules per mol per kelvin?"},
    {"domain": "physics_engineering", "q": "What is the acceleration due to gravity on Earth's surface in m/s²?"},
    {"domain": "physics_engineering", "q": "What is the speed of sound in air at 20°C in m/s?"},
    {"domain": "physics_engineering", "q": "What is the speed of sound in water in m/s?"},
    {"domain": "physics_engineering", "q": "What is the speed of sound in steel in m/s?"},
    {"domain": "physics_engineering", "q": "What is the density of air at sea level in kg/m³?"},
    {"domain": "physics_engineering", "q": "What is the density of water at 4°C in kg/m³?"},
    {"domain": "physics_engineering", "q": "What is the density of steel in kg/m³?"},
    {"domain": "physics_engineering", "q": "What is the density of aluminum in kg/m³?"},
    {"domain": "physics_engineering", "q": "What is the density of gold in kg/m³?"},
    {"domain": "physics_engineering", "q": "What is the density of iron in kg/m³?"},
    {"domain": "physics_engineering", "q": "What is the density of copper in kg/m³?"},
    {"domain": "physics_engineering", "q": "What is the density of lead in kg/m³?"},
    {"domain": "physics_engineering", "q": "What is the density of glass in kg/m³?"},
    {"domain": "physics_engineering", "q": "What is the density of concrete in kg/m³?"},
    {"domain": "physics_engineering", "q": "What is the density of bone in kg/m³?"},
    {"domain": "physics_engineering", "q": "What is the density of human fat tissue in kg/m³?"},
    {"domain": "physics_engineering", "q": "What is the density of nitrogen gas at STP in kg/m³?"},
    {"domain": "physics_engineering", "q": "What is the density of carbon dioxide gas at STP in kg/m³?"},
    {"domain": "physics_engineering", "q": "What is the specific heat capacity of water in J/kg/K?"},
    {"domain": "physics_engineering", "q": "What is the specific heat capacity of air in J/kg/K?"},
    {"domain": "physics_engineering", "q": "What is the latent heat of vaporization of water in J/kg?"},
    {"domain": "physics_engineering", "q": "What is the latent heat of fusion of water in J/kg?"},
    {"domain": "physics_engineering", "q": "What is the Young's modulus of steel in pascals?"},
    {"domain": "physics_engineering", "q": "What is the refractive index of glass?"},
    {"domain": "physics_engineering", "q": "What is the refractive index of water?"},
    {"domain": "physics_engineering", "q": "What is the electrical resistivity of copper in ohm-meters?"},
    {"domain": "physics_engineering", "q": "What is the thermal conductivity of steel in W/m/K?"},
    {"domain": "physics_engineering", "q": "What is the energy of one photon of visible light (550 nm) in joules?"},
    {"domain": "physics_engineering", "q": "What is the kinetic energy of a car (1500 kg) at highway speed (100 km/h) in joules?"},
    {"domain": "physics_engineering", "q": "What is the power output of a typical hair dryer in watts?"},
    {"domain": "physics_engineering", "q": "What is the power output of a typical incandescent light bulb in watts?"},
    {"domain": "physics_engineering", "q": "What is the power output of a nuclear power plant in watts?"},
    {"domain": "physics_engineering", "q": "What is the energy in one kilowatt-hour in joules?"},
    {"domain": "physics_engineering", "q": "What is the energy released by one ton of TNT in joules?"},
    {"domain": "physics_engineering", "q": "What is the energy content of one liter of gasoline in joules?"},
    {"domain": "physics_engineering", "q": "What is the voltage of a standard AA battery in volts?"},
    {"domain": "physics_engineering", "q": "How many watts does an average household in the US consume?"},
    {"domain": "physics_engineering", "q": "What is the mass of a loaded semi-truck in kilograms?"},
    {"domain": "physics_engineering", "q": "What is the mass of a Boeing 747 aircraft fully loaded in kilograms?"},
    {"domain": "physics_engineering", "q": "What is the mass of a typical car in kilograms?"},
    {"domain": "physics_engineering", "q": "What is the length of a Boeing 747 in meters?"},
    {"domain": "physics_engineering", "q": "What is the wingspan of a Boeing 747 in meters?"},
    {"domain": "physics_engineering", "q": "What is the length of a standard shipping container in meters?"},
    {"domain": "physics_engineering", "q": "What is the height of a standard shipping container in meters?"},
    {"domain": "physics_engineering", "q": "What is the diameter of a human hair in meters?"},
    {"domain": "physics_engineering", "q": "What is the diameter of a red blood cell in meters?"},
    {"domain": "physics_engineering", "q": "What is the diameter of a typical bacterium in meters?"},
    {"domain": "physics_engineering", "q": "What is the diameter of a typical virus in meters?"},
    {"domain": "physics_engineering", "q": "What is the diameter of a DNA double helix in nanometers?"},
    {"domain": "physics_engineering", "q": "What is the length of a furlong in meters?"},
    {"domain": "physics_engineering", "q": "How many feet are in one furlong?"},
    {"domain": "physics_engineering", "q": "What is the length of one nautical mile in meters?"},
    {"domain": "physics_engineering", "q": "How many inches are in one foot?"},
    {"domain": "physics_engineering", "q": "How many feet are in one mile?"},
    {"domain": "physics_engineering", "q": "How many meters are in one mile?"},
    {"domain": "physics_engineering", "q": "How many liters are in one cubic meter?"},
    {"domain": "physics_engineering", "q": "How many milliliters are in one liter?"},
    {"domain": "physics_engineering", "q": "How many cubic centimeters are in one liter?"},
    {"domain": "physics_engineering", "q": "How many gallons are in one cubic meter?"},
    {"domain": "physics_engineering", "q": "How many pounds are in one kilogram?"},
    {"domain": "physics_engineering", "q": "How many ounces are in one pound?"},
    {"domain": "physics_engineering", "q": "What is the pressure at the bottom of the Mariana Trench in pascals?"},
    {"domain": "physics_engineering", "q": "What is atmospheric pressure at sea level in pascals?"},
    {"domain": "physics_engineering", "q": "What is the volume of an Olympic swimming pool in cubic meters?"},
    {"domain": "physics_engineering", "q": "What is the volume of a standard basketball in liters?"},
    {"domain": "physics_engineering", "q": "What is the diameter of a standard basketball in meters?"},
    {"domain": "physics_engineering", "q": "What is the diameter of a baseball in meters?"},
    {"domain": "physics_engineering", "q": "What is the mass of a baseball in grams?"},
    {"domain": "physics_engineering", "q": "What is the diameter of a golf ball in meters?"},
    {"domain": "physics_engineering", "q": "What is the mass of a golf ball in grams?"},
    {"domain": "physics_engineering", "q": "What is the diameter of a tennis ball in meters?"},
    {"domain": "physics_engineering", "q": "What is the length of a standard football field in meters?"},
    {"domain": "physics_engineering", "q": "What is the data storage capacity of a standard DVD in gigabytes?"},
    {"domain": "physics_engineering", "q": "What is the data storage capacity of a Blu-ray disc in gigabytes?"},
    {"domain": "physics_engineering", "q": "What is the data storage capacity of a standard CD in megabytes?"},
    {"domain": "physics_engineering", "q": "What is the data storage capacity of a floppy disk in kilobytes?"},
    {"domain": "physics_engineering", "q": "What is the clock speed of a modern CPU in GHz?"},
    {"domain": "physics_engineering", "q": "How many transistors are in a modern CPU?"},
    {"domain": "physics_engineering", "q": "What is the peak performance of the fastest supercomputer in floating point operations per second?"},
    {"domain": "physics_engineering", "q": "What is the size of one bit in terms of physical storage on a hard drive?"},
    {"domain": "physics_engineering", "q": "What is the mass of a standard brick in kilograms?"},
    {"domain": "physics_engineering", "q": "What is the thickness of a standard sheet of paper in millimeters?"},
    {"domain": "physics_engineering", "q": "What is the mass of a sheet of A4 paper in grams?"},
    {"domain": "physics_engineering", "q": "What is the length of a standard toothpick in meters?"},
    {"domain": "physics_engineering", "q": "What is the diameter of a grain of sand in millimeters?"},
    {"domain": "physics_engineering", "q": "What is the diameter of a human egg cell in meters?"},
    {"domain": "physics_engineering", "q": "What is the length of a football field including end zones in meters?"},
    {"domain": "physics_engineering", "q": "What is the volume of a Mason jar in liters?"},
    {"domain": "physics_engineering", "q": "What is the horsepower of a typical passenger car engine?"},
    {"domain": "physics_engineering", "q": "What is the thrust of a Boeing 747 engine in newtons?"},
    {"domain": "physics_engineering", "q": "What is the energy needed to lift 1 kg by 1 meter on Earth in joules?"},
    {"domain": "physics_engineering", "q": "What is the wavelength of visible light (green) in nanometers?"},
    {"domain": "physics_engineering", "q": "What is the frequency of middle C on a piano in hertz?"},

    # -----------------------------------------------------------------------
    # BIOLOGY & MEDICINE (128)
    # -----------------------------------------------------------------------
    {"domain": "biology_medicine", "q": "What is the average mass of an adult human in kilograms?"},
    {"domain": "biology_medicine", "q": "What is the average height of an adult human in meters?"},
    {"domain": "biology_medicine", "q": "How many cells are in the human body?"},
    {"domain": "biology_medicine", "q": "How many hairs does the average human have on their head?"},
    {"domain": "biology_medicine", "q": "What is the diameter of a human hair in micrometers?"},
    {"domain": "biology_medicine", "q": "What is the length of the human genome in base pairs?"},
    {"domain": "biology_medicine", "q": "What is the length of DNA in a single human cell in meters?"},
    {"domain": "biology_medicine", "q": "How many neurons are in the human brain?"},
    {"domain": "biology_medicine", "q": "What is the volume of the human brain in liters?"},
    {"domain": "biology_medicine", "q": "What is the average heart rate of a resting adult in beats per minute?"},
    {"domain": "biology_medicine", "q": "How many times does the human heart beat in a year?"},
    {"domain": "biology_medicine", "q": "What is the volume of blood in the human body in liters?"},
    {"domain": "biology_medicine", "q": "What is the diameter of a red blood cell in micrometers?"},
    {"domain": "biology_medicine", "q": "How many red blood cells are in one milliliter of blood?"},
    {"domain": "biology_medicine", "q": "What is the mass of a red blood cell in kilograms?"},
    {"domain": "biology_medicine", "q": "What is the average lifespan of a red blood cell in days?"},
    {"domain": "biology_medicine", "q": "What is the average lung capacity of an adult in liters?"},
    {"domain": "biology_medicine", "q": "How many breaths does a human take per minute at rest?"},
    {"domain": "biology_medicine", "q": "What is the volume of air inhaled per breath in liters?"},
    {"domain": "biology_medicine", "q": "What is the length of the human small intestine in meters?"},
    {"domain": "biology_medicine", "q": "What is the length of the human large intestine in meters?"},
    {"domain": "biology_medicine", "q": "How many bones are in the adult human body?"},
    {"domain": "biology_medicine", "q": "What is the mass of the human skeleton in kilograms?"},
    {"domain": "biology_medicine", "q": "What is the average number of calories burned per day by an adult human?"},
    {"domain": "biology_medicine", "q": "How many calories are in one gram of fat?"},
    {"domain": "biology_medicine", "q": "How many calories are in one gram of carbohydrate?"},
    {"domain": "biology_medicine", "q": "How many calories are in one gram of protein?"},
    {"domain": "biology_medicine", "q": "How many calories are in one gram of sugar?"},
    {"domain": "biology_medicine", "q": "How many calories are in a Mars Bar?"},
    {"domain": "biology_medicine", "q": "How long does it take the human digestive system to digest a meal in hours?"},
    {"domain": "biology_medicine", "q": "How many liters of water does a human drink per day on average?"},
    {"domain": "biology_medicine", "q": "What is the speed of a nerve impulse in the human body in m/s?"},
    {"domain": "biology_medicine", "q": "How many synapses are in the human brain?"},
    {"domain": "biology_medicine", "q": "What is the diameter of a typical human cell in micrometers?"},
    {"domain": "biology_medicine", "q": "What is the volume of a typical human cell in cubic meters?"},
    {"domain": "biology_medicine", "q": "What is the mass of a typical human cell in kilograms?"},
    {"domain": "biology_medicine", "q": "How many mitochondria are in a typical human cell?"},
    {"domain": "biology_medicine", "q": "What is the diameter of a mitochondrion in micrometers?"},
    {"domain": "biology_medicine", "q": "What is the diameter of the nucleus of a human cell in micrometers?"},
    {"domain": "biology_medicine", "q": "How many base pairs are in one gene on average?"},
    {"domain": "biology_medicine", "q": "How many genes are in the human genome?"},
    {"domain": "biology_medicine", "q": "What is the average number of words in the Harry Potter book series?"},
    {"domain": "biology_medicine", "q": "What is the number of characters in the human genome if written as text?"},
    {"domain": "biology_medicine", "q": "What is the average mass of a blue whale in kilograms?"},
    {"domain": "biology_medicine", "q": "What is the length of a blue whale in meters?"},
    {"domain": "biology_medicine", "q": "What is the mass of an African elephant in kilograms?"},
    {"domain": "biology_medicine", "q": "What is the mass of a domestic cat in kilograms?"},
    {"domain": "biology_medicine", "q": "What is the mass of a honeybee in grams?"},
    {"domain": "biology_medicine", "q": "What is the mass of an ant in milligrams?"},
    {"domain": "biology_medicine", "q": "How many ants are in an average ant colony?"},
    {"domain": "biology_medicine", "q": "What is the length of a typical ant in millimeters?"},
    {"domain": "biology_medicine", "q": "How many hairs are on a black bear?"},
    {"domain": "biology_medicine", "q": "What is the length of a typical bacterium in micrometers?"},
    {"domain": "biology_medicine", "q": "What is the mass of a typical bacterium in kilograms?"},
    {"domain": "biology_medicine", "q": "How many bacteria are in the human gut?"},
    {"domain": "biology_medicine", "q": "What is the size of a typical virus in nanometers?"},
    {"domain": "biology_medicine", "q": "What is the diameter of an E. coli bacterium in micrometers?"},
    {"domain": "biology_medicine", "q": "What is the length of a human chromosome in meters?"},
    {"domain": "biology_medicine", "q": "How many chromosomes does a human cell have?"},
    {"domain": "biology_medicine", "q": "What is the size of a buckminsterfullerene molecule (C60) in nanometers?"},
    {"domain": "biology_medicine", "q": "What is the mass of a single E. coli cell in kilograms?"},
    {"domain": "biology_medicine", "q": "What is the average lifespan of a human in years?"},
    {"domain": "biology_medicine", "q": "What is the average number of steps a human takes per day?"},
    {"domain": "biology_medicine", "q": "What is the diameter of a human egg cell in micrometers?"},
    {"domain": "biology_medicine", "q": "What is the length of a human sperm cell in micrometers?"},
    {"domain": "biology_medicine", "q": "How many sperm cells are in a typical ejaculation?"},
    {"domain": "biology_medicine", "q": "What is the mass of the human liver in kilograms?"},
    {"domain": "biology_medicine", "q": "What is the mass of the human heart in grams?"},
    {"domain": "biology_medicine", "q": "What is the mass of the human brain in kilograms?"},
    {"domain": "biology_medicine", "q": "What is the surface area of the human lung alveoli in square meters?"},
    {"domain": "biology_medicine", "q": "What is the total length of capillaries in the human body in kilometers?"},
    {"domain": "biology_medicine", "q": "What is the diameter of a human capillary in micrometers?"},
    {"domain": "biology_medicine", "q": "How many people in the US have sickle cell anemia?"},
    {"domain": "biology_medicine", "q": "What is the number of people diagnosed with cancer in the US per year?"},
    {"domain": "biology_medicine", "q": "What is the average speed of a running human in m/s?"},
    {"domain": "biology_medicine", "q": "What is the maximum recorded speed of a human sprinter in m/s?"},
    {"domain": "biology_medicine", "q": "What is the swimming speed of a blue whale in m/s?"},
    {"domain": "biology_medicine", "q": "What is the flight speed of a common bird (pigeon) in m/s?"},
    {"domain": "biology_medicine", "q": "What is the average number of hairs on a human eyebrow?"},
    {"domain": "biology_medicine", "q": "What is the average duration of a human blink in milliseconds?"},
    {"domain": "biology_medicine", "q": "What is the thickness of human skin in millimeters?"},
    {"domain": "biology_medicine", "q": "What is the total surface area of human skin in square meters?"},
    {"domain": "biology_medicine", "q": "What is the mass of a chicken egg in grams?"},
    {"domain": "biology_medicine", "q": "What is the mass of an ostrich egg in kilograms?"},
    {"domain": "biology_medicine", "q": "What is the mass of a grain of rice in milligrams?"},
    {"domain": "biology_medicine", "q": "How many grapes does it take to make one kilogram of raisins?"},
    {"domain": "biology_medicine", "q": "What is the mass of a standard Hershey Bar in grams?"},
    {"domain": "biology_medicine", "q": "What is the number of hotdogs consumed in the US per year?"},
    {"domain": "biology_medicine", "q": "How many pounds of added sugar does the average American consume per year?"},
    {"domain": "biology_medicine", "q": "How many liters of blood does the human heart pump per minute?"},
    {"domain": "biology_medicine", "q": "What is the thickness of a human fingernail in millimeters?"},
    {"domain": "biology_medicine", "q": "How fast do human fingernails grow in mm per month?"},
    {"domain": "biology_medicine", "q": "What is the number of taste buds on the human tongue?"},
    {"domain": "biology_medicine", "q": "What is the force of a human bite in newtons?"},
    {"domain": "biology_medicine", "q": "What is the average body temperature of a human in Celsius?"},
    {"domain": "biology_medicine", "q": "How many ATP molecules does a human cell produce per day?"},
    {"domain": "biology_medicine", "q": "What is the speed of a nerve impulse in a myelinated neuron in m/s?"},
    {"domain": "biology_medicine", "q": "How many olfactory receptors does a human have?"},
    {"domain": "biology_medicine", "q": "What is the length of an average human nerve cell in meters?"},
    {"domain": "biology_medicine", "q": "How many photoreceptor cells are in the human eye?"},
    {"domain": "biology_medicine", "q": "What is the number of blood donations made per year in the US?"},
    {"domain": "biology_medicine", "q": "How many liters of urine does a human produce per day?"},
    {"domain": "biology_medicine", "q": "What is the volume of a human stomach in liters?"},
    {"domain": "biology_medicine", "q": "What is the circumference of the Earth at the equator in meters?"},
    {"domain": "biology_medicine", "q": "How many blue whales placed side by side equal the circumference of Earth?"},
    {"domain": "biology_medicine", "q": "What is the mass of a proton relative to an electron?"},
    {"domain": "biology_medicine", "q": "How many base pairs fit in one millimeter of DNA?"},
    {"domain": "biology_medicine", "q": "What is the mass of the human body's total DNA in grams?"},
    {"domain": "biology_medicine", "q": "What is the average mass of a human baby at birth in kilograms?"},
    {"domain": "biology_medicine", "q": "What is the size of a SO (Thiomargarita) bacterium in micrometers?"},
    {"domain": "biology_medicine", "q": "How many neurons does the human spinal cord contain?"},
    {"domain": "biology_medicine", "q": "What is the diameter of the human aorta in centimeters?"},
    {"domain": "biology_medicine", "q": "What is the average speed of peristalsis in the human digestive tract in cm/s?"},
    {"domain": "biology_medicine", "q": "What is the surface area of the human small intestine in square meters?"},
    {"domain": "biology_medicine", "q": "What is the mass of a human kidney in grams?"},
    {"domain": "biology_medicine", "q": "How long does it take for food to travel through the human digestive system in hours?"},
    {"domain": "biology_medicine", "q": "What is the total number of red blood cells produced per day in the human body?"},
    {"domain": "biology_medicine", "q": "What is the diameter of a human platelet in micrometers?"},
    {"domain": "biology_medicine", "q": "How many white blood cells are in one milliliter of human blood?"},
    {"domain": "biology_medicine", "q": "What is the average length of a human life in seconds?"},
    {"domain": "biology_medicine", "q": "What is the mass of a human thumb in grams?"},
    {"domain": "biology_medicine", "q": "What fraction of the human body is water by mass?"},
    {"domain": "biology_medicine", "q": "How many cells does the human body replace per day?"},

    # -----------------------------------------------------------------------
    # EARTH SCIENCE (63)
    # -----------------------------------------------------------------------
    {"domain": "earth_science", "q": "What is the radius of the Earth in meters?"},
    {"domain": "earth_science", "q": "What is the mass of the Earth in kilograms?"},
    {"domain": "earth_science", "q": "What is the circumference of the Earth at the equator in kilometers?"},
    {"domain": "earth_science", "q": "What is the surface area of the Earth in square meters?"},
    {"domain": "earth_science", "q": "What is the total volume of the Earth's oceans in cubic meters?"},
    {"domain": "earth_science", "q": "What is the total mass of the Earth's oceans in kilograms?"},
    {"domain": "earth_science", "q": "What is the average depth of the ocean in meters?"},
    {"domain": "earth_science", "q": "What is the depth of the Mariana Trench in meters?"},
    {"domain": "earth_science", "q": "What is the total land area of Earth in square kilometers?"},
    {"domain": "earth_science", "q": "What is the area of the United States in square kilometers?"},
    {"domain": "earth_science", "q": "What is the area of the continent of Africa in square kilometers?"},
    {"domain": "earth_science", "q": "What is the height of Mount Everest in meters?"},
    {"domain": "earth_science", "q": "What is the height of Mount Aconcagua in meters?"},
    {"domain": "earth_science", "q": "What is the length of the Amazon River in kilometers?"},
    {"domain": "earth_science", "q": "What is the length of the Nile River in kilometers?"},
    {"domain": "earth_science", "q": "What is the volume of the Amazon River discharge per second in cubic meters?"},
    {"domain": "earth_science", "q": "What is the average depth of the Mariana Trench in meters?"},
    {"domain": "earth_science", "q": "What is the total mass of the atmosphere of Earth in kilograms?"},
    {"domain": "earth_science", "q": "What is the height of the troposphere in kilometers?"},
    {"domain": "earth_science", "q": "What is the concentration of CO2 in the atmosphere in parts per million?"},
    {"domain": "earth_science", "q": "What is the total number of grains of sand on Earth?"},
    {"domain": "earth_science", "q": "What is the diameter of a grain of coarse sand in millimeters?"},
    {"domain": "earth_science", "q": "What is the diameter of a grain of fine sand in millimeters?"},
    {"domain": "earth_science", "q": "What is the total volume of glaciers and ice caps on Earth in cubic meters?"},
    {"domain": "earth_science", "q": "What is the mass of the Greenland ice sheet in kilograms?"},
    {"domain": "earth_science", "q": "What is the average temperature of the Earth's surface in Celsius?"},
    {"domain": "earth_science", "q": "What is the temperature at the center of the Earth in Celsius?"},
    {"domain": "earth_science", "q": "What is the pressure at the center of the Earth in pascals?"},
    {"domain": "earth_science", "q": "What is the density at the center of the Earth in kg/m³?"},
    {"domain": "earth_science", "q": "What is the age of the Earth in years?"},
    {"domain": "earth_science", "q": "What is the total amount of freshwater on Earth as a fraction of all water?"},
    {"domain": "earth_science", "q": "What is the volume of Lake Superior in cubic kilometers?"},
    {"domain": "earth_science", "q": "What is the length of the longest waterfall (Angel Falls) in meters?"},
    {"domain": "earth_science", "q": "What is the length of the Olo'upena Falls in meters?"},
    {"domain": "earth_science", "q": "What is the area of the Sahara Desert in square kilometers?"},
    {"domain": "earth_science", "q": "What is the speed of Earth's rotation at the equator in m/s?"},
    {"domain": "earth_science", "q": "What is the speed of Earth's orbit around the Sun in km/s?"},
    {"domain": "earth_science", "q": "What is the total annual rainfall on Earth in cubic meters per year?"},
    {"domain": "earth_science", "q": "What is the area of the Amazon rainforest in square kilometers?"},
    {"domain": "earth_science", "q": "What is the total number of trees on Earth?"},
    {"domain": "earth_science", "q": "What is the thickness of Earth's crust in kilometers?"},
    {"domain": "earth_science", "q": "What is the total energy released by all earthquakes per year in joules?"},
    {"domain": "earth_science", "q": "What is the energy released by the largest recorded earthquake in joules?"},
    {"domain": "earth_science", "q": "What is the area of Disney World in square kilometers?"},
    {"domain": "earth_science", "q": "What is the area of London Heathrow Airport in square kilometers?"},
    {"domain": "earth_science", "q": "What is the area of Hilton Head Island in square kilometers?"},
    {"domain": "earth_science", "q": "What is the total annual freshwater consumption by humans in cubic meters?"},
    {"domain": "earth_science", "q": "What is the salinity of seawater in grams per kilogram?"},
    {"domain": "earth_science", "q": "What is the speed of the Gulf Stream current in m/s?"},
    {"domain": "earth_science", "q": "What is the energy released by a typical hurricane per day in joules?"},
    {"domain": "earth_science", "q": "What is the total mass of all living organisms on Earth in kilograms?"},
    {"domain": "earth_science", "q": "What is the area of a typical swimming pool in square meters?"},
    {"domain": "earth_science", "q": "What is the volume of the Great Pyramid of Giza in cubic meters?"},
    {"domain": "earth_science", "q": "What is the height of the Eiffel Tower in meters?"},
    {"domain": "earth_science", "q": "What is the length of the Great Wall of China in kilometers?"},
    {"domain": "earth_science", "q": "What is the depth of the deepest mine in meters?"},
    {"domain": "earth_science", "q": "What is the area of the Pacific Ocean in square kilometers?"},
    {"domain": "earth_science", "q": "What is the total mass of plastic waste produced globally per year in kilograms?"},
    {"domain": "earth_science", "q": "What is the rate of sea level rise per year in millimeters?"},
    {"domain": "earth_science", "q": "What is the volume of water evaporated from oceans per year in cubic meters?"},
    {"domain": "earth_science", "q": "What is the radius of the Earth's inner core in kilometers?"},
    {"domain": "earth_science", "q": "What is the mass of the Himalayas mountain range in kilograms?"},
    {"domain": "earth_science", "q": "What is the total length of all rivers on Earth in kilometers?"},

    # -----------------------------------------------------------------------
    # ASTRONOMY (38)
    # -----------------------------------------------------------------------
    {"domain": "astronomy", "q": "What is the mass of the Sun in kilograms?"},
    {"domain": "astronomy", "q": "What is the radius of the Sun in meters?"},
    {"domain": "astronomy", "q": "What is the luminosity of the Sun in watts?"},
    {"domain": "astronomy", "q": "What is the surface temperature of the Sun in Kelvin?"},
    {"domain": "astronomy", "q": "What is the temperature at the core of the Sun in Kelvin?"},
    {"domain": "astronomy", "q": "What is the distance from Earth to the Sun in meters?"},
    {"domain": "astronomy", "q": "What is the distance from Earth to the Moon in meters?"},
    {"domain": "astronomy", "q": "What is the mass of the Moon in kilograms?"},
    {"domain": "astronomy", "q": "What is the radius of the Moon in meters?"},
    {"domain": "astronomy", "q": "What is the orbital period of the Moon around Earth in days?"},
    {"domain": "astronomy", "q": "What is the diameter of the observable universe in meters?"},
    {"domain": "astronomy", "q": "What is the age of the universe in years?"},
    {"domain": "astronomy", "q": "What is the distance to the nearest star (Proxima Centauri) in light-years?"},
    {"domain": "astronomy", "q": "What is the diameter of the Milky Way galaxy in light-years?"},
    {"domain": "astronomy", "q": "How many stars are in the Milky Way galaxy?"},
    {"domain": "astronomy", "q": "What is the distance from Earth to the center of the Milky Way in light-years?"},
    {"domain": "astronomy", "q": "What is the orbital period of Earth around the Sun in days?"},
    {"domain": "astronomy", "q": "What is the orbital speed of Earth around the Sun in km/s?"},
    {"domain": "astronomy", "q": "What is the mass of Jupiter in kilograms?"},
    {"domain": "astronomy", "q": "What is the radius of Jupiter in meters?"},
    {"domain": "astronomy", "q": "What is the mass of Mars in kilograms?"},
    {"domain": "astronomy", "q": "What is the surface gravity on Mars in m/s²?"},
    {"domain": "astronomy", "q": "What is the distance from the Sun to Neptune in meters?"},
    {"domain": "astronomy", "q": "What is the escape velocity from Earth in m/s?"},
    {"domain": "astronomy", "q": "What is the escape velocity from the Sun in m/s?"},
    {"domain": "astronomy", "q": "What is the orbital period of Halley's Comet in years?"},
    {"domain": "astronomy", "q": "What is the diameter of Halley's Comet nucleus in kilometers?"},
    {"domain": "astronomy", "q": "What is the number of galaxies in the observable universe?"},
    {"domain": "astronomy", "q": "What is the Hubble constant in km/s/Mpc?"},
    {"domain": "astronomy", "q": "What is the critical density of the universe in kg/m³?"},
    {"domain": "astronomy", "q": "What is the distance one light-year equals in meters?"},
    {"domain": "astronomy", "q": "What is the distance one parsec equals in light-years?"},
    {"domain": "astronomy", "q": "What is the angular diameter of the Sun as seen from Earth in degrees?"},
    {"domain": "astronomy", "q": "What is the probability of a total solar eclipse at any given location per year?"},
    {"domain": "astronomy", "q": "What is the diameter of the Sun relative to Earth?"},
    {"domain": "astronomy", "q": "What is the volume of the Sun in cubic meters?"},
    {"domain": "astronomy", "q": "What is the density of the Sun in kg/m³?"},
    {"domain": "astronomy", "q": "How many Earth masses is Jupiter?"},

    # -----------------------------------------------------------------------
    # CHEMISTRY (36)
    # -----------------------------------------------------------------------
    {"domain": "chemistry", "q": "What is the molar mass of water (H2O) in grams per mol?"},
    {"domain": "chemistry", "q": "What is the molar mass of carbon dioxide (CO2) in grams per mol?"},
    {"domain": "chemistry", "q": "What is the molar mass of oxygen (O2) in grams per mol?"},
    {"domain": "chemistry", "q": "What is the molar mass of nitrogen (N2) in grams per mol?"},
    {"domain": "chemistry", "q": "What is the molar mass of iron in grams per mol?"},
    {"domain": "chemistry", "q": "What is the molar mass of gold in grams per mol?"},
    {"domain": "chemistry", "q": "What is the molar mass of carbon in grams per mol?"},
    {"domain": "chemistry", "q": "What is the molar mass of sodium chloride (NaCl) in grams per mol?"},
    {"domain": "chemistry", "q": "What is the molar mass of glucose (C6H12O6) in grams per mol?"},
    {"domain": "chemistry", "q": "What is the molar mass of ethanol in grams per mol?"},
    {"domain": "chemistry", "q": "What is the density of ethanol in kg/m³?"},
    {"domain": "chemistry", "q": "What is the density of oxygen gas at STP in kg/m³?"},
    {"domain": "chemistry", "q": "What is the density of hydrogen gas at STP in kg/m³?"},
    {"domain": "chemistry", "q": "What is the volume of one mole of ideal gas at STP in liters?"},
    {"domain": "chemistry", "q": "How many molecules of water are in 18 mL of water?"},
    {"domain": "chemistry", "q": "What is the diameter of a water molecule in nanometers?"},
    {"domain": "chemistry", "q": "What is the diameter of a carbon atom in nanometers?"},
    {"domain": "chemistry", "q": "What is the atomic mass of carbon-12 in kg?"},
    {"domain": "chemistry", "q": "What is the atomic radius of a hydrogen atom in nanometers?"},
    {"domain": "chemistry", "q": "What is the bond length of a C-C bond in nanometers?"},
    {"domain": "chemistry", "q": "What is the enthalpy of combustion of methane in kJ/mol?"},
    {"domain": "chemistry", "q": "What is the enthalpy of combustion of glucose in kJ/mol?"},
    {"domain": "chemistry", "q": "What is the pH of pure water at 25°C?"},
    {"domain": "chemistry", "q": "What is the boiling point of water at 1 atm in Celsius?"},
    {"domain": "chemistry", "q": "What is the melting point of iron in Celsius?"},
    {"domain": "chemistry", "q": "What is the melting point of gold in Celsius?"},
    {"domain": "chemistry", "q": "What is the fraction by volume of oxygen in Earth's atmosphere?"},
    {"domain": "chemistry", "q": "What is the fraction by volume of nitrogen in Earth's atmosphere?"},
    {"domain": "chemistry", "q": "How many electrons are in a garbage can full of air?"},
    {"domain": "chemistry", "q": "What is the volume fraction of TiO2 in a gallon of paint?"},
    {"domain": "chemistry", "q": "What is the molar volume of an ideal gas at room temperature (25°C) in liters?"},
    {"domain": "chemistry", "q": "What is the binding energy of a hydrogen atom in electron volts?"},
    {"domain": "chemistry", "q": "What is the half-life of carbon-14 in years?"},
    {"domain": "chemistry", "q": "What is the half-life of uranium-235 in years?"},
    {"domain": "chemistry", "q": "What is the energy released by fission of one uranium-235 nucleus in joules?"},
    {"domain": "chemistry", "q": "What is the molar mass of titanium dioxide (TiO2) in grams per mol?"},

    # -----------------------------------------------------------------------
    # MATHEMATICS (30)
    # -----------------------------------------------------------------------
    {"domain": "mathematics", "q": "What is the value of pi to 10 decimal places?"},
    {"domain": "mathematics", "q": "What is the value of e (Euler's number) to 10 decimal places?"},
    {"domain": "mathematics", "q": "What is the value of the golden ratio phi?"},
    {"domain": "mathematics", "q": "What is log10(2)?"},
    {"domain": "mathematics", "q": "What is log10(3)?"},
    {"domain": "mathematics", "q": "What is ln(2)?"},
    {"domain": "mathematics", "q": "What is pi squared?"},
    {"domain": "mathematics", "q": "What is the value of pi to the power of pi?"},
    {"domain": "mathematics", "q": "What is 10^10^10?"},
    {"domain": "mathematics", "q": "What is 7 to the power of 43?"},
    {"domain": "mathematics", "q": "How many seconds are in one year?"},
    {"domain": "mathematics", "q": "How many seconds are in one day?"},
    {"domain": "mathematics", "q": "How many days are in one year?"},
    {"domain": "mathematics", "q": "How many hours are in one year?"},
    {"domain": "mathematics", "q": "What is the factorial of 20?"},
    {"domain": "mathematics", "q": "What is the number of ways to shuffle a standard deck of 52 cards?"},
    {"domain": "mathematics", "q": "What is the sum of the prime factors of 12043242?"},
    {"domain": "mathematics", "q": "What is the square root of 2?"},
    {"domain": "mathematics", "q": "What is the cube root of 1000000?"},
    {"domain": "mathematics", "q": "What is 2 to the power of 100?"},
    {"domain": "mathematics", "q": "What is 10 to the power of 100 (googol)?"},
    {"domain": "mathematics", "q": "What is the value of 1 radian in degrees?"},
    {"domain": "mathematics", "q": "What is the surface area formula for a sphere of radius r?"},
    {"domain": "mathematics", "q": "What is the volume formula for a sphere of radius r?"},
    {"domain": "mathematics", "q": "What is the probability of getting heads 10 times in a row with a fair coin?"},
    {"domain": "mathematics", "q": "What is the number of prime numbers less than 1000?"},
    {"domain": "mathematics", "q": "What is the value of sqrt(100 + sqrt(100 + sqrt(100)^(-7))^(-8))^(-9)?"},
    {"domain": "mathematics", "q": "What is 1 divided by pi squared?"},
    {"domain": "mathematics", "q": "What is the value of (1/(pi^pi))^2?"},
    {"domain": "mathematics", "q": "What is Euler's identity (e^(i*pi) + 1)?"},

    # -----------------------------------------------------------------------
    # OTHER SCIENCE / EVERYDAY / SOCIAL (94)
    # -----------------------------------------------------------------------
    {"domain": "other", "q": "What is the population of the United States?"},
    {"domain": "other", "q": "What is the population of the world?"},
    {"domain": "other", "q": "What is the population of New York City?"},
    {"domain": "other", "q": "What is the population of Boston, Massachusetts?"},
    {"domain": "other", "q": "What is the population of Singapore?"},
    {"domain": "other", "q": "What is the population of the United Kingdom?"},
    {"domain": "other", "q": "What is the population of China?"},
    {"domain": "other", "q": "What is the population of India?"},
    {"domain": "other", "q": "How many cars are registered in the United States?"},
    {"domain": "other", "q": "How many smartphones are in use globally?"},
    {"domain": "other", "q": "How many cell phone calls were made in the US in 2003?"},
    {"domain": "other", "q": "How many people in the US use cell phones?"},
    {"domain": "other", "q": "How many basketballs are sold in the US each year?"},
    {"domain": "other", "q": "How many piano tuners are in Chicago?"},
    {"domain": "other", "q": "How many gas stations are in the United States?"},
    {"domain": "other", "q": "How many people visit Disney World per year?"},
    {"domain": "other", "q": "What is the capacity of a standard school bus in passengers?"},
    {"domain": "other", "q": "What is the capacity of a standard elevator in kilograms?"},
    {"domain": "other", "q": "What is the average annual income of a person in the US in dollars?"},
    {"domain": "other", "q": "What is the GDP of the United States in dollars?"},
    {"domain": "other", "q": "What is Bill Gates' net worth in US dollars?"},
    {"domain": "other", "q": "How many words are in the English language?"},
    {"domain": "other", "q": "How many words does the average person speak per day?"},
    {"domain": "other", "q": "How many books are published in the US per year?"},
    {"domain": "other", "q": "What is the number of pages in a typical novel?"},
    {"domain": "other", "q": "How many copies of Harry Potter and the Sorcerer's Stone have been sold worldwide?"},
    {"domain": "other", "q": "How many words are in the Harry Potter book series?"},
    {"domain": "other", "q": "What is the number of characters in one page of a book?"},
    {"domain": "other", "q": "How many songs has Taylor Swift released?"},
    {"domain": "other", "q": "How many ex-lovers does Taylor Swift reference in Blank Space?"},
    {"domain": "other", "q": "How long is a typical Super Mario Bros speedrun in seconds?"},
    {"domain": "other", "q": "What is the frame rate of the original Super Mario Bros game in fps?"},
    {"domain": "other", "q": "What is the length of Toy Story 4 movie in minutes?"},
    {"domain": "other", "q": "How many computers did Pixar use to render Toy Story 4?"},
    {"domain": "other", "q": "What is the number of floating point operations in one frame of Toy Story 4?"},
    {"domain": "other", "q": "What is the render time per frame for a Pixar movie in CPU hours?"},
    {"domain": "other", "q": "How much does a Sonic Chili Cheese Coney hot dog weigh in grams?"},
    {"domain": "other", "q": "How many calories are in a Sonic Chili Cheese Coney?"},
    {"domain": "other", "q": "What is the mass of a King Size Hershey Bar in grams?"},
    {"domain": "other", "q": "What is the volume of a standard garbage can in liters?"},
    {"domain": "other", "q": "What is the mass of a standard garbage can full of air in kilograms?"},
    {"domain": "other", "q": "What is the volume of a standard school room in cubic meters?"},
    {"domain": "other", "q": "What is the length of a standard school room in meters?"},
    {"domain": "other", "q": "What is the capacity of an Olympic swimming pool in liters?"},
    {"domain": "other", "q": "How many people donate blood in the US per year?"},
    {"domain": "other", "q": "What is the volume of blood donated per donation in liters?"},
    {"domain": "other", "q": "How many swimming pools can be filled with US annual blood donations?"},
    {"domain": "other", "q": "What is the nuclear material needed to power 10 US homes in kilograms?"},
    {"domain": "other", "q": "How much nuclear waste does a typical reactor produce per year in kilograms?"},
    {"domain": "other", "q": "How many hours of video are uploaded to YouTube per minute?"},
    {"domain": "other", "q": "What is the total data stored on the internet in exabytes?"},
    {"domain": "other", "q": "How many emails are sent per day worldwide?"},
    {"domain": "other", "q": "What is the number of active websites on the internet?"},
    {"domain": "other", "q": "What is the length of the MIT infinite corridor in meters?"},
    {"domain": "other", "q": "What is the height of the Golden 1 Center stadium in Sacramento in meters?"},
    {"domain": "other", "q": "What is the capacity of the Golden 1 Center stadium in Sacramento?"},
    {"domain": "other", "q": "What is the length of an aircraft carrier (USS Nimitz) in meters?"},
    {"domain": "other", "q": "What is the mass of an aircraft carrier (USS Nimitz) in kilograms?"},
    {"domain": "other", "q": "What is the mass of HMS Queen Elizabeth aircraft carrier in kilograms?"},
    {"domain": "other", "q": "What is the capacity of an average aircraft carrier in number of aircraft?"},
    {"domain": "other", "q": "How many cars can fit in a standard parking garage?"},
    {"domain": "other", "q": "What is the height of a standard floor in a building in meters?"},
    {"domain": "other", "q": "What is the length of a standard car in meters?"},
    {"domain": "other", "q": "What is the height of a standard car in meters?"},
    {"domain": "other", "q": "What is the average fuel economy of a US car in miles per gallon?"},
    {"domain": "other", "q": "How many gallons of gasoline are consumed per day in the US?"},
    {"domain": "other", "q": "What is the number of commercial flights per day worldwide?"},
    {"domain": "other", "q": "What is the mass of a cubic meter of seawater in kilograms?"},
    {"domain": "other", "q": "What is the thickness of a credit card in millimeters?"},
    {"domain": "other", "q": "What is the size of a standard car sunshield in square meters?"},
    {"domain": "other", "q": "How many car sunshields laid flat would cover Earth's atmosphere at 100 km altitude?"},
    {"domain": "other", "q": "How many standard sunshields fit side by side around Earth's circumference at 100 km?"},
    {"domain": "other", "q": "What is the total number of books ever written?"},
    {"domain": "other", "q": "What is the storage size of one page of text in bytes?"},
    {"domain": "other", "q": "What is the storage size of one book in megabytes?"},
    {"domain": "other", "q": "What is the storage capacity of an iPod classic in gigabytes?"},
    {"domain": "other", "q": "What is the data storage ratio between a Blu-ray and a CD?"},
    {"domain": "other", "q": "What is the number of Mozy cloud storage units in one iPod?"},
    {"domain": "other", "q": "What is the mass of a standard Person (average human) in kg?"},
    {"domain": "other", "q": "How many average humans make up 503 kg?"},
    {"domain": "other", "q": "What is the area of Delhi, India in square kilometers?"},
    {"domain": "other", "q": "What is the number of Squid Game players in season 1?"},
    {"domain": "other", "q": "What is the probability of winning the Squid Game glass bridge with 16 players?"},
    {"domain": "other", "q": "How many seconds does the average person spend on their phone per day?"},
    {"domain": "other", "q": "How many ping pong balls fit in a Boeing 747?"},
    {"domain": "other", "q": "What is the diameter of a ping pong ball in meters?"},
    {"domain": "other", "q": "What is the interior volume of a Boeing 747 in cubic meters?"},
    {"domain": "other", "q": "How many piano tuners does New York City need?"},
    {"domain": "other", "q": "How many golf balls fit in a school bus?"},
    {"domain": "other", "q": "What is the mass of the internet in kilograms?"},
    {"domain": "other", "q": "How many people in the US have internet access?"},
    {"domain": "other", "q": "What is the number of Google searches per day worldwide?"},
    {"domain": "other", "q": "What is the annual revenue of Google in US dollars?"},
    {"domain": "other", "q": "How many words per minute does an average person type?"},
    {"domain": "other", "q": "How long would it take to type out the human genome at 60 words per minute?"},

    # top-up to reach 500
    {"domain": "physics_engineering", "q": "What is the half-life of a muon in microseconds?"},
    {"domain": "biology_medicine", "q": "What is the average number of eyelashes on a human eye?"},
    {"domain": "biology_medicine", "q": "What is the average number of hairs on a human forearm?"},
    {"domain": "earth_science", "q": "What is the total length of coastline of the United States in kilometers?"},
    {"domain": "astronomy", "q": "What is the diameter of Pluto in kilometers?"},
    {"domain": "chemistry", "q": "What is the boiling point of nitrogen in Celsius?"},
    {"domain": "mathematics", "q": "What is the value of the Euler-Mascheroni constant gamma?"},
    {"domain": "other", "q": "What is the average price of a gallon of milk in the US in dollars?"},
    {"domain": "other", "q": "How many public schools are there in the United States?"},
]

def fetch_tavily(question: str, api_key: str, timeout: int = 15) -> str:
    """Call Tavily once and return the best snippet as a string."""
    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "query": question,
            "search_depth": "basic",
            "topic": "general",
            "max_results": 2,
            "include_answer": False,
            "include_raw_content": False,
        },
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    # return best snippet from results
    for r in data.get("results", []):
        snippet = (r.get("content") or r.get("snippet") or "").strip()
        if snippet:
            return snippet[:500]
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print questions and domain counts, no Tavily calls")
    ap.add_argument("--output", default=str(OUTPUT))
    ap.add_argument("--resume", action="store_true",
                    help="Skip questions already in output file")
    args = ap.parse_args()

    from collections import Counter
    counts = Counter(q["domain"] for q in ASSUMPTION_QUESTIONS)
    print(f"Total questions: {len(ASSUMPTION_QUESTIONS)}")
    for d, c in counts.most_common():
        print(f"  {d:<25} {c}")

    if args.dry_run:
        print("\n--dry-run set, no Tavily calls made.")
        return

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        sys.exit("TAVILY_API_KEY not set. Run: source .env")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if resuming
    existing: dict[str, dict] = {}
    if args.resume and output_path.exists():
        with output_path.open() as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} already fetched.")

    results: dict[str, dict] = dict(existing)
    errors = 0

    for i, item in enumerate(ASSUMPTION_QUESTIONS):
        q = item["q"]
        if q in results:
            continue  # already fetched

        try:
            snippet = fetch_tavily(q, api_key)
            results[q] = {"domain": item["domain"], "question": q, "answer": snippet}
            status = "ok" if snippet else "empty"
        except Exception as e:
            results[q] = {"domain": item["domain"], "question": q, "answer": "", "error": str(e)}
            status = f"ERROR: {e}"
            errors += 1

        done = len(results)
        print(f"[{done:3d}/500] [{status}] {q[:70]}")

        # Save after every 10 fetches
        if done % 10 == 0:
            with output_path.open("w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(0.3)  # gentle rate limiting

    # Final save
    with output_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    filled = sum(1 for v in results.values() if v.get("answer"))
    print(f"\nDone. {filled}/500 answered, {errors} errors → {output_path}")


if __name__ == "__main__":
    main()
