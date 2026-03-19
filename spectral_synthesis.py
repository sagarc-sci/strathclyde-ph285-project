import numpy, scipy, pandas
import itertools, logging, sys
from collections.abc import Callable
from functools import cache


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
if not LOG.hasHandlers():
    LOG.addHandler(logging.StreamHandler(sys.stderr))


class Particle(object):
    '''
        Model a charged particle that can interact with photons
        
        Interactions are modelled by adding transitions that can move the particle
        to a different excitation or ionisation
    '''
    def __init__(self, sybmol: str, protons: int, quantum_number: int, charge: int):
        self.symbol: str = sybmol
        self.protons: int = protons
        self.quantum_number: int = quantum_number
        self.charge: int = charge
        self.outgoing_transitions: set['Transition'] = set()
        self.incoming_transitions: set['Transition'] = set()
    
    def __repr__(self) -> str:
        return f'[{self.symbol}]'

    def add_outgoing_transition(self, transition: 'Transition') -> 'Particle':
        self.outgoing_transitions.add(transition)
        return self

    def add_incoming_transition(self, transition: 'Transition') -> 'Particle':
        self.incoming_transitions.add(transition)
        return self
    
    @cache
    def ground_state(self) -> 'Particle':
        if 1 == self.quantum_number:
            return self
        else:
            excitation = self
            while (excitation := excitation.previous_excitation()):
                if 1 == excitation.quantum_number:
                    return excitation
        return None
    
    @cache
    def previous_excitation(self) -> 'Particle':
        return max((transition.source for transition in self.incoming_transitions if isinstance(transition, BoundBoundAbsorption)),
                   key=lambda particle: particle.quantum_number, default=None)

    @cache
    def next_excitation(self) -> 'Particle':
        return min((transition.destination for transition in self.outgoing_transitions if isinstance(transition, BoundBoundAbsorption)),
                   key=lambda particle: particle.quantum_number, default=None)
    
    @cache
    def elemental_state(self) -> 'Particle':
        if 0 == self.charge:
            return self
        elif self.charge > 0:
            next_function = Particle.previous_ion
        else:
            next_function = Particle.next_ion
        ion = self
        while (ion := next_function(ion)):
            if 0 == ion.charge:
                return ion
        return None

    @cache
    def previous_ion(self) -> 'Particle':
        for transition in self.incoming_transitions:
            if isinstance(transition, BoundFreeAbsorption):
                return transition.source
        return None
    
    @cache
    def next_ion(self) -> 'Particle':
        for transition in self.outgoing_transitions:
            if isinstance(transition, BoundFreeAbsorption):
                return transition.destination
        return None
    
    @cache
    def degeneracy(self) -> float:
        '''
            Returns degeneracy of quantum number describing the current state

            For Hydrogenic species with a single electron described using quantum number n,
            degeneracy is 2 * n^2
        '''
        # Only true for Hydrogenic species
        return 2 * self.quantum_number ** 2
    
    @cache
    def ionisation_energy(self) -> float:
        '''
            Returns ionisation energy of this particle

            This is the same as negative of electron energy state
            For Hydrogenic species with single electron,
            ionisation energy, X = R_inf * (Z^2 / n^2)
        '''
        # Only true for Hydrogenic species
        return scipy.constants.physical_constants['Rydberg constant times hc in J'][0] * (self.protons / self.quantum_number) ** 2

    @cache
    def ionisation_wavelength(self) -> float:
        return (scipy.constants.h * scipy.constants.c) / self.ionisation_energy()
    
    @cache
    def partition_function(self, temperature: float) -> float:
        if self.protons - self.charge == 0:
            return 1.
        return self.degeneracy() * numpy.exp((self.ionisation_energy() - self.ground_state().ionisation_energy())
                                             / (scipy.constants.k * temperature))


class Transition(object):
    def __init__(self, source: 'Particle', destination: 'Particle'):
        self.source: 'Particle' = source
        self.destination: 'Particle' = destination

        self.source.add_outgoing_transition(self)
        self.destination.add_incoming_transition(self)
    
    def symbol(self) -> str:
        return NotImplemented
    
    def __repr__(self) -> str:
        return f'{self.source} -- {self.symbol()} --> {self.destination}'
    
    def cross_section(self, wavelength: float|numpy.ndarray, cell: 'Cell') -> float|numpy.ndarray:
        return NotImplemented
    
    def opacity(self, wavelength: float|numpy.ndarray, cell: 'Cell') -> float|numpy.ndarray:
        return cell.particle_number_densities[self.source] * self.cross_section(wavelength, cell)


class ThomsonScattering(Transition):
    def symbol(self):
        return 'TH'
    
    def cross_section(self, wavelength: float|numpy.ndarray, cell: 'Cell') -> float|numpy.ndarray:
        return 6.65e-29 # m2


class FreeFreeAbsorption(Transition):
    CONSTANT_TERM: float = ((numpy.sqrt(32 * numpy.pi) * scipy.constants.e ** 6)
                     / ((3 * numpy.sqrt(3) * scipy.constants.h * scipy.constants.c ** 4)
                        * numpy.sqrt(scipy.constants.k * scipy.constants.m_e ** 3)))

    def symbol(self) -> str:
        return 'FF'

    def gaunt_factor(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        # Quantum mechanical correction factor
        # Not implemented
        return 1.

    def cross_section(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        # Only true for Hydrogenic species
        return (self.CONSTANT_TERM
                * (wavelength ** 3 / numpy.sqrt(cell.temperature))
                * sum(particle_number_density * particle.protons ** 2
                      for particle, particle_number_density in cell.particle_number_densities.items()
                      if particle.charge != 0 and particle is not cell.atmosphere.electron)
                * self.gaunt_factor(wavelength, cell))


class BoundFreeAbsorption(Transition):
    def __init__(self, source: 'Particle', destination: 'Particle'):
        super().__init__(source, destination)
        self.constant_term: float = (((64 * (numpy.pi ** 4) * (scipy.constants.e ** 10) * scipy.constants.m_e)
                                      / (3 * numpy.sqrt(3) * (scipy.constants.c ** 4) * (scipy.constants.h ** 6)
                                         * ((4 * numpy.pi * scipy.constants.epsilon_0) ** 5)))
                                         * (self.source.protons ** 4 / self.source.quantum_number ** 5))

    def symbol(self) -> str:
        return 'BF'

    def gaunt_factor(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        # Quantum mechanical correction factor
        # Not implemented
        return 1.

    def cross_section(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        # Only true for Hydrogenic species
        return numpy.where(wavelength <= self.source.ionisation_wavelength(),
                           self.constant_term * (wavelength ** 3) * self.gaunt_factor(wavelength, cell),
                           0.)


class BoundBoundAbsorption(Transition):
    def __init__(self, source: 'Particle', destination: 'Particle', oscillator_strength: float):
        super().__init__(source, destination)
        self.line_wavelength: float = 1 / ((1 / self.source.ionisation_wavelength()) - (1 / self.destination.ionisation_wavelength()))
        self.line_frequency: float = scipy.constants.c / self.line_wavelength
        self.cross_section_constant_term: float = ((scipy.constants.e ** 2 / (4
                                                                              * scipy.constants.epsilon_0
                                                                              * scipy.constants.m_e
                                                                              * scipy.constants.c))
                                                                              * oscillator_strength)
        self.doppler_width_constant_term: float = (numpy.sqrt(2 * scipy.constants.k / (scipy.constants.m_p * self.source.protons))
                                                   / self.line_wavelength)
                                 
    def symbol(self) -> str:
        return 'BB'
    
    def doppler_profile(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        doppler_width = self.doppler_width_constant_term * numpy.sqrt(cell.temperature)
        frequency = scipy.constants.c / wavelength
        return numpy.exp(- ((frequency - self.line_frequency) / doppler_width) ** 2) / (doppler_width * numpy.sqrt(numpy.pi))
    
    def line_profile(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        return self.doppler_profile(wavelength, cell)

    def gaunt_factor(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        # Quantum mechanical correction factor
        # Not implemented
        return 1.

    def cross_section(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        return (self.cross_section_constant_term
                * (1 - ((self.source.degeneracy() * cell.particle_number_densities[self.destination])
                        / (self.destination.degeneracy() * cell.particle_number_densities[self.source])))
                * self.line_profile(wavelength, cell)
                * self.gaunt_factor(wavelength, cell))


class OscillatorStrength(object):
    def __init__(self, filename):
        self.dataset = pandas.read_csv(filename)

    def value(self, protons: int, charge: int, quantum_number: int, excited_quantum_number: int):
        query = ((self.dataset['protons'] == protons)
                 & (self.dataset['charge'] == charge)
                 & (self.dataset['quantum_number'] == quantum_number)
                 & (self.dataset['excited_quantum_number'] == excited_quantum_number))
        result = self.dataset.loc[query]
        if result.size > 0:
            return self.dataset.at[result.index[0], 'oscillator_strength']
        else:
            # Quantum mechanical correction factor
            # Not implemented
            gaunt_factor = 1.
            # Menzel-Pekeris Approximation
            # Only true for Hydrogenic species
            return (((32 / (3 * numpy.pi * numpy.sqrt(3))) / (quantum_number ** 5 * excited_quantum_number ** 3))
                    * ((1 / quantum_number ** 2) - (1 / excited_quantum_number ** 2)) ** (-3)
                    * gaunt_factor)


class Cell(object):
    def opacity(self, wavelength: float|numpy.ndarray) -> float|numpy.ndarray:
        return NotImplemented
    
    def __mul__(self, other):
        # Convinience method to invoke opacity on a vector of wavelengths
        # without having to vectorise the Cell matrix
        if isinstance(other, (float, numpy.floating, numpy.ndarray)):
            return self.opacity(other)
        return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)


class EmptyCell(Cell):
    def opacity(self, wavelength: float|numpy.ndarray) -> float|numpy.ndarray:
        return 0.


class DenseCell(Cell):
    def __init__(self, atmosphere: 'Atmosphere', temperature: float, number_density: float, elemental_composition: dict['Particle', float]):
        self.atmosphere: 'Atmosphere' = atmosphere
        self.temperature: float = temperature
        self.number_density: float = number_density
        self.elemental_composition: dict['Particle', float] = elemental_composition
        self.particle_number_densities: dict['Particle', float] = self.solve_particle_number_densities()

    def solve_particle_number_densities(self) -> dict['Particle', float]:
        particle_number_densities: dict['Particle', float] = dict()

        ion_number_densities, electron_number_density = self.solve_ion_number_densities()

        particle_number_densities[self.atmosphere.electron] = electron_number_density

        for ion, ion_number_density in ion_number_densities.items():
            particle_number_densities.update(self.solve_excitation_number_densities(ion, ion_number_density))
        
        return particle_number_densities

    def solve_ion_number_densities(self, max_electron_density_error: float=0.01, max_iterations: int=100) -> tuple[dict['Particle', float], float]:
        ions_of_element: dict['Particle', list['Particle']] = dict()
        for element in self.elemental_composition:
            ions = set([element])
            for next_function in [Particle.previous_ion, Particle.next_ion]:
                ion = element
                while (ion := next_function(ion)):
                    ion = ion.ground_state()
                    ions.add(ion)
            ions_of_element[element] = sorted(ions, key=lambda ion: ion.charge)

        partition_function_values: dict['Particle', float] = dict()
        for ions in ions_of_element.values():
            for ion in ions:
                partition_function_values[ion] = ion.partition_function(self.temperature)
                excitation = ion
                while(excitation := excitation.next_excitation()):
                    partition_function_values[ion] += excitation.partition_function(self.temperature)

        ion_number_densities: dict['Particle', float] = \
            dict((ion, self.elemental_composition[element] * self.number_density / len(ions))
                 for element, ions in ions_of_element.items()
                 for ion in ions)
        electron_number_density = 0.
        has_electron_density_converged = False
        constant_term = 2 * ((2 * numpy.pi * scipy.constants.m_e * scipy.constants.k) ** 1.5) / (scipy.constants.h ** 3)
        for iteration in range(max_iterations):
            updated_electron_number_density = sum(ion.charge * ion_number_density
                                                  for ion, ion_number_density in ion_number_densities.items())
            if numpy.isclose(electron_number_density, updated_electron_number_density, rtol=max_electron_density_error):
                electron_number_density = updated_electron_number_density
                has_electron_density_converged = True
                break
            electron_number_density = updated_electron_number_density
            for element, ions in ions_of_element.items():
                for i in range(1, len(ions)):
                    ion_number_densities[ions[i]] = ((constant_term
                     * (partition_function_values[ions[i]] / partition_function_values[ions[i - 1]])
                     * (self.temperature ** 1.5)
                     * numpy.exp(- ions[i - 1].ionisation_energy() / (scipy.constants.k * self.temperature)))
                     * (ion_number_densities[ions[i - 1]] / electron_number_density))
                scaling_factor = (self.elemental_composition[element] * self.number_density
                                  / sum(ion_number_densities[ion] for ion in ions))
                for ion in ions:
                    ion_number_densities[ion] *= scaling_factor
        if not has_electron_density_converged:
            LOG.warning('Electron number density did not converge')

        return ion_number_densities, electron_number_density
    
    def solve_excitation_number_densities(self, particle: 'Particle', particle_number_density: float) -> dict['Particle', float]:
        partition_function_values: dict['Particle', float] = dict()
        partition_function_values[particle] = particle.partition_function(self.temperature)
        excitation = particle
        while (excitation := excitation.next_excitation()):
            partition_function_values[excitation] = excitation.partition_function(self.temperature)
        scaling_factor = particle_number_density / sum(partition_function_values.values())

        excitation_number_densities: dict['Particle', float] = \
            dict((excitation, partition_function_value * scaling_factor)
                 for excitation, partition_function_value in partition_function_values.items())
        
        return excitation_number_densities

    def opacity(self, wavelength: float|numpy.ndarray) -> float|numpy.ndarray:
        return sum(transition.opacity(wavelength, self) for transition in self.atmosphere.transitions)


class Atmosphere(object):
    def __init__(self,
                 electron: 'Particle',
                 particles: set['Particle'],
                 transitions: set['Transition'],
                 elemental_composition: dict['Particle', float],
                 thickness: float,
                 density_gradient: Callable[[float|numpy.ndarray], float|numpy.ndarray],
                 temperature_gradient: Callable[[float|numpy.ndarray], float|numpy.ndarray],
                 core_density: float=None,
                 surface_density: float=None,
                 core_temperature: float=None,
                 surface_temperature: float=None):
        self.electron: 'Particle' = electron
        self.particles: set['Particle'] = particles
        self.transitions: set['Transition'] = transitions
        self.elemental_composition: dict['Particle', float] = elemental_composition
        self.thickness: float = thickness
        self.density_gradient: Callable[[float|numpy.ndarray], float|numpy.ndarray] = density_gradient
        self.temperature_gradient: Callable[[float|numpy.ndarray], float|numpy.ndarray] = temperature_gradient
        self.core_density: float = core_density
        self.surface_density: float = surface_density
        self.core_temperature: float = core_temperature
        self.surface_temperature: float = surface_temperature
    
    def cells(self, grid_size: float) -> numpy.ndarray:
        cell_positions = numpy.arange(0, self.thickness, grid_size) + (grid_size / 2)

        temperature_delta = self.temperature_gradient(cell_positions) * grid_size
        cumulative_temperature_delta = numpy.cumsum(temperature_delta)
        if self.core_temperature is not None:
            cell_temperatures = self.core_temperature + cumulative_temperature_delta
        else:
            cell_temperatures = self.surface_temperature - cumulative_temperature_delta

        density_delta = self.density_gradient(cell_positions) * grid_size
        cumulative_density_delta = numpy.cumsum(density_delta)
        if self.core_density is not None:
            cell_densities = self.core_density + cumulative_density_delta
        else:
            cell_densities = self.surface_density - cumulative_density_delta

        return numpy.array([DenseCell(self, cell_temperatures[i], cell_densities[i], self.elemental_composition)
                for i in range(cell_positions.size)])


class Source(object):
    def photons(self, sample_size: int) -> numpy.ndarray:
        return NotImplemented


class BlackBodySource(Source):
    PLANCK_RADIANCE_CONSTANT_TERM: float = 2 * scipy.constants.h * scipy.constants.c ** 2
    WIEN_DISPLACEMENT_CONSTANT_TERM: float = 2.897e-3

    def __init__(self, temperature: float, wavelength_bounding_box_scaling_factors: tuple[float, float]=(0.1, 8), excess: float=5.):
        self.temperature: float = temperature
        self.wavelength_bounding_box_scaling_factors: tuple[float, float] = wavelength_bounding_box_scaling_factors
        self.excess: float = excess

        self.planck_radiance_exponent_constant_term: float = numpy.exp(scipy.constants.h * scipy.constants.c / (scipy.constants.k * temperature))

        # Wien's Displacement Law
        self.wavelength_at_max_radiance: float = self.WIEN_DISPLACEMENT_CONSTANT_TERM / temperature
        self.max_spectral_radiance: float = self.spectral_radiance(self.wavelength_at_max_radiance)

    def spectral_radiance(self, wavelength: float) -> float:
        return (self.PLANCK_RADIANCE_CONSTANT_TERM / (wavelength ** 5 * ((self.planck_radiance_exponent_constant_term ** (1 / wavelength)) - 1)))

    def photons(self, sample_size: int) -> numpy.ndarray:
        sample_size_with_excess = int(sample_size * self.excess)
        wavelengths = numpy.random.uniform(*[limit * self.wavelength_at_max_radiance for limit in self.wavelength_bounding_box_scaling_factors],
                                           sample_size_with_excess)
        radiance = numpy.random.uniform(0, self.max_spectral_radiance, sample_size_with_excess)
        allowed_radiance = self.spectral_radiance(wavelengths)
        allowed_wavelengths = wavelengths[radiance < allowed_radiance]
        LOG.info(f'Sampled {sample_size_with_excess} photons; accepted {allowed_wavelengths.size}; required {sample_size}')
        return allowed_wavelengths[0:sample_size]


class SphericalVolumetricSource(Source):
    def __init__(self, source: 'Source', half_angular_span: float):
        self.source: 'Source' = source
        self.half_angular_span: float = half_angular_span
    
    def photons(self, sample_size: int) -> numpy.ndarray:
        wavelengths = self.source.photons(sample_size)
        angular_direction = numpy.random.uniform(-self.half_angular_span, self.half_angular_span, wavelengths.shape)
        return numpy.vstack((wavelengths,
                             numpy.cos(angular_direction),
                             numpy.sin(angular_direction))).T


class Geometry(object):
    EMPTY_CELL = EmptyCell()

    def __init__(self, source: 'Source', source_span: float, atmosphere: 'Atmosphere', grid_size: float):
        self.source: 'Source' = source
        self.source_span: float = source_span
        self.atmosphere: 'Atmosphere' = atmosphere
        self.grid_size: float = grid_size

        self.atmosphere_cells = self.atmosphere.cells(self.grid_size)

    def positions(self, photons: numpy.ndarray, steps: int|numpy.ndarray=None, previous_positions: float|numpy.ndarray=None) -> float|numpy.ndarray:
        return NotImplemented
    
    def cells(self, positions: float|numpy.ndarray) -> Cell|numpy.ndarray:
        return NotImplemented
    

class PlanarGeometry(Geometry):
    def positions(self, photons: numpy.ndarray, steps: int|numpy.ndarray=None, previous_positions: float|numpy.ndarray=None) -> float|numpy.ndarray:
        return self.source_span + self.grid_size * steps if steps is not None else previous_positions + self.grid_size
    
    def cells(self, positions: float|numpy.ndarray) -> Cell|numpy.ndarray:
        return numpy.where((positions < self.source_span) & (positions > (self.source_span + self.atmosphere.thickness)),
                           self.EMPTY_CELL,
                           self.atmosphere_cells[numpy.asarray((positions - self.source_span)/self.grid_size, dtype=int)])


class SphericalGeometry(Geometry):
    def __init__(self, source, source_span, atmosphere, grid_size):
        self.source_half_angular_span = numpy.arcsin(source_span / (source_span + atmosphere.thickness))

        super().__init__(SphericalVolumetricSource(source, self.source_half_angular_span), source_span, atmosphere, grid_size)

    def positions(self, photons: numpy.ndarray, steps: int|numpy.ndarray=None, previous_positions: numpy.ndarray=None) -> numpy.ndarray:
        direction_vectors = photons[:, 1:]
        
        if steps is not None:
            initial_vectors = direction_vectors[:, 1::-1]
            upper_hemisphere_vectors = initial_vectors[:, 0] < 0
            initial_vectors[upper_hemisphere_vectors][0] *= -1
            initial_vectors[~upper_hemisphere_vectors][1] *= -1
            return self.source_span * initial_vectors + (self.grid_size * steps) * direction_vectors
        else:
            return previous_positions + self.grid_size * direction_vectors
    
    def cells(self, positions: numpy.ndarray) -> Cell|numpy.ndarray:
        radial_distances = numpy.linalg.norm(positions, axis=-1)
        return numpy.where((radial_distances < self.source_span) & (radial_distances > (self.source_span + self.atmosphere.thickness)),
                           self.EMPTY_CELL,
                           self.atmosphere_cells[numpy.asarray((radial_distances - self.source_span)/self.grid_size, dtype=int)])


if __name__ == '__main__':
    import argparse, os, json
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--file-prefix')
    parser.add_argument('-x', '--skip-simulation', action='store_true')
    args = parser.parse_args()

    from matplotlib import pyplot

    # Setup default figure size and use LaTeX for text formatting
    pyplot.rc('figure', figsize=[12, 8], dpi=144)
    pyplot.rc('text', usetex=True)

    # Load run configuration
    with open(f'{args.file_prefix}-config.json', 'r') as fyle:
        run_config = json.load(fyle)

    if not args.skip_simulation:
        # Load oscillator strength lookup table
        oscillator_strength = OscillatorStrength('oscillator_strengths.csv')

        # Setup a pure Hydrogen atmosphere and light-matter interactions
        electron = Particle('e', 0, 1, -1)
        particles = set()
        transitions = set([ThomsonScattering(electron, electron), FreeFreeAbsorption(electron, electron)])

        hydrogen_excitations = [Particle(f'H{i}', 1, i, 0) for i in range(1, 11)]
        hydrogen_ion = Particle('H+', 1, 1, 1)

        for particle in itertools.chain(hydrogen_excitations, [hydrogen_ion]):
            particles.add(particle)

        for i in range(len(hydrogen_excitations) - 1):
            for j in range(i + 1, len(hydrogen_excitations)):
                transitions.add(BoundBoundAbsorption(hydrogen_excitations[i],
                                                    hydrogen_excitations[j],
                                                    oscillator_strength.value(hydrogen_excitations[i].protons,
                                                                                hydrogen_excitations[i].charge,
                                                                                hydrogen_excitations[i].quantum_number,
                                                                                hydrogen_excitations[j].quantum_number)))
        for excitation in hydrogen_excitations:
            transitions.add(BoundFreeAbsorption(excitation, hydrogen_ion))
        
        # Setup temperature and density gradients and initialise atmosphere
        zero_gradient = lambda r: numpy.zeros_like(r) if isinstance(r, numpy.ndarray) else 0.
        atmosphere_thickness = run_config["atmosphere"]["thickness"]
        grid_size = 0.003
        atmosphere = Atmosphere(electron, particles, transitions, {hydrogen_excitations[0]: 1.},
                                atmosphere_thickness, zero_gradient, zero_gradient,
                                core_density=run_config["atmosphere"]["core_density"],
                                core_temperature=run_config["atmosphere"]["core_temperature"])
        
        # Setup a Black Body radiation source with same temperature as at inner boundary of atmosphere
        source = BlackBodySource(run_config["source"]["temperature"])

        # Setup grid geometry
        if run_config["geometry"]["type"] == "spherical":
            # Spherical atmosphere
            geometry = SphericalGeometry(source, 0.66, atmosphere, grid_size)
            steps = numpy.arange(0, int(atmosphere_thickness/grid_size) + 1, 1)[:, numpy.newaxis, numpy.newaxis]
        else:
            # Plane parallel atmosphere
            geometry = PlanarGeometry(source, 0.66, atmosphere, grid_size)
            steps = numpy.arange(0, int(atmosphere_thickness/grid_size) + 1, 1)[:, numpy.newaxis]

        # Begin Monte-Carlo simulation

        # Generate photons
        input_photons = geometry.source.photons(run_config["source"]["photons"])

        # Handle previously chosen geometry
        is_directional_photon = len(input_photons.shape) > 1

        if is_directional_photon:
            input_wavelengths = input_photons[:, 0]
            input_directions = input_photons[:, 2] # Extract sine of angular direction
        else:
            input_wavelengths = input_photons
            input_directions = None
        
        # Save input photons
        numpy.save(f'{args.file_prefix}-input-wavelengths.npy', input_wavelengths)
        if is_directional_photon:
            numpy.save(f'{args.file_prefix}-input-directions.npy', input_directions)

        # For each photon calculate the position vector at each step
        # Result: (step, photon, co-ordinates)
        positions = geometry.positions(input_photons, steps=steps)

        # Map the position vectors to grid cells
        # Result: (step, photon, cell)
        cells = geometry.cells(positions)

        # For each wavelength calculate opacity it experiences at that step
        # Result: (step, photon, opacity)
        opacities = cells * input_wavelengths

        # Sample probabilities of absorption for each photon at each step
        # Result: (step, photon, absorption probability)
        # Collapse along step axis to check if photon is absorbed at any step
        # Result: (photon, is absorbed?)
        is_absorbed = numpy.any(numpy.random.random(opacities.shape) < opacities, axis=0)

        # Gather surviving photons
        output_photons = input_photons[~is_absorbed]

        # Extract surviving wavelengths (and directions at origin)
        if is_directional_photon:
            output_wavelengths = output_photons[:, 0]
            output_directions = output_photons[:, 2]
        else:
            output_wavelengths = output_photons
            output_directions = None

        # Save output photons
        numpy.save(f'{args.file_prefix}-output-wavelengths.npy', output_wavelengths)
        if is_directional_photon:
            numpy.save(f'{args.file_prefix}-output-directions.npy', output_directions)
    else:
        is_directional_photon = False
        input_directions = None
        output_directions = None
        input_wavelengths = numpy.load(f'{args.file_prefix}-input-wavelengths.npy')
        output_wavelengths = numpy.load(f'{args.file_prefix}-output-wavelengths.npy')
        if os.path.exists(f'{args.file_prefix}-input-directions.npy'):
            is_directional_photon = True
            input_directions = numpy.load(f'{args.file_prefix}-input-directions.npy')
            output_directions = numpy.load(f'{args.file_prefix}-output-directions.npy')

    # Analyse results and plot

    # Plot the emission spectrum
    pyplot.figure(1)
    pyplot.hist(input_wavelengths, 1000, label=r'$Black~Body~Spectrum$')

    # Overlay the absorption spectrum
    pyplot.hist(output_wavelengths, 1000, label=r'$Atmospheric~Spectrum$')
    pyplot.legend()
    pyplot.show()

    if is_directional_photon:
        # Plot directional intensities from source
        pyplot.figure(2)
        pyplot.hist(input_directions, 1000, label=r'$Source$')

        # Overlay directional intensities past atmosphere
        pyplot.hist(output_directions, 1000, label=r'$Atmosphere$')
        pyplot.legend()
        pyplot.show()

    # Obtain and plot relative intensities
    def relative_intensity(input_signal: numpy.ndarray, output_signal: numpy.ndarray, bins: int=1000) -> tuple[numpy.ndarray, numpy.ndarray]:
        input_intensity, bin_edges = numpy.histogram(input_signal, bins=bins)
        output_intensity, output_bin_edges = numpy.histogram(output_signal, bins=bin_edges)
        intensity_ratio = output_intensity / input_intensity
        intensity_ratio[numpy.isnan(intensity_ratio)] = 1.
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, intensity_ratio
    
    # Plot relative intensity at each wavelength to demonstrate absorption in atmosphere
    wavelength_bins, relative_intensity_at_wavelength = relative_intensity(input_wavelengths, output_wavelengths)
    pyplot.figure(3)
    pyplot.plot(wavelength_bins * 1e9, relative_intensity_at_wavelength)
    pyplot.show()

    if is_directional_photon:
        # Plot relative intensity in each direction to demonstrate Limb Darkening
        direction_bins, relative_intensity_in_direction = relative_intensity(input_directions, output_directions)
        pyplot.figure(4)
        pyplot.plot(direction_bins, relative_intensity_in_direction)
        pyplot.show()
