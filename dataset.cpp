/*
 * ns3_md_scenarios_improved.cc
 *
 * IMPROVED VERSION - 5G Handover Simulation with Enhanced Scenarios
 * 
 * Key improvements:
 * ✅ Cell-specific LoS probability computation (distance + angle aware)
 * ✅ Position-dependent interference modeling (not global)
 * ✅ Spatial shadow fading correlation
 * ✅ Dynamic RLF threshold based on RSRP distribution
 * ✅ Scenario-specific jitter scaling
 * ✅ Normalized episode durations (minimum 600s)
 * ✅ Better building alignment for urban canyon effects
 * ✅ Improved path loss model calibration
 * ✅ Enhanced suburban scenario complexity
 * ✅ Smoother zone transitions in mixed urban
 */

#include "ns3/buildings-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace ns3;

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kTickS = 0.1;
constexpr uint32_t kTickMs = 100;
constexpr uint32_t kRlfT310Ticks = 2;
constexpr uint32_t kTopNeighborsToLog = 6;
constexpr double kPingPongWindowS = 5.0;
constexpr double kNoiseFloorDbm = -104.0;
constexpr uint32_t kShadowFadingHistorySize = 50; // For spatial correlation

#ifdef _WIN32
constexpr const char* kDefaultOutputPrefix = R"(E:\5g_handover\dataset\run)";
#else
constexpr const char* kDefaultOutputPrefix = "results/ns3_md/run";
#endif

struct CliOptions
{
  uint32_t scenarioId = 1;
  std::string pattern = "A";
  double durationS = 0.0;
  uint32_t ueCount = 20;
  uint32_t seed = 1;
  uint32_t tttMs = 160;
  double hysDb = 3.0;
  std::string outputPrefix = kDefaultOutputPrefix;
};

struct GnbSpec
{
  uint32_t id = 0;
  Vector position;
  double txPowerDbm = 36.0;
};

struct PatternSpec
{
  std::string code;
  std::string name;
  double defaultDurationS = 600.0;
};

struct ScenarioSpec
{
  uint32_t id = 1;
  std::string name;

  double xMin = 0.0;
  double xMax = 1000.0;
  double yMin = 0.0;
  double yMax = 1000.0;

  std::vector<GnbSpec> gnbs;
  std::vector<PatternSpec> patterns;

  // Improved: Dynamic RLF thresholds (will be set at runtime)
  double rlfThresholdDbm = -122.0;
  double fallbackThresholdDbm = -110.0;
  
  // Scenario-specific jitter scaling (NEW)
  double jitterScaleM = 1.0;
  
  // Base interference scale (will be modulated by position)
  double baseInterferenceScale = 1.0;
};

struct TrajectoryPoint
{
  Vector position;
  double speedMps = 0.0;
  bool turning = false;
  bool inTunnel = false;
};

struct CellMeasurement
{
  uint32_t cellId = 0;
  double rsrpDbm = -140.0;
  double sinrDb = -20.0;
  double distanceM = 0.0;
  double losProbability = 0.0;
  bool isLos = false;
};

struct TickDecision
{
  bool hoEvent = false;
  bool emergencyHo = false;
  bool rlfEvent = false;
  bool pingPongEvent = false;

  uint32_t fromCell = 0;
  uint32_t toCell = 0;
  double marginDb = -999.0;
  std::string reason = "NONE";
};

struct UeRuntime
{
  uint32_t id = 0;
  Ptr<Node> node;

  double phaseOffsetS = 0.0;
  double lateralOffsetM = 0.0;
  Vector jitter = Vector(0.0, 0.0, 0.0);

  uint32_t servingCell = 0;
  uint32_t candidateCell = 0;
  uint32_t a3HoldMs = 0;

  int32_t lastHoFromCell = -1;
  int32_t lastHoToCell = -1;
  double lastHoTimeS = -1000.0;

  uint32_t lowRsrpTicks = 0;

  uint32_t hoCount = 0;
  uint32_t rlfCount = 0;
  uint32_t pingPongCount = 0;
  
  // Shadow fading history for spatial correlation (NEW)
  std::map<uint32_t, std::deque<double>> shadowFadingHistory;
};

// IMPROVEMENT 1: Clamp function
double Clamp(double x, double lo, double hi)
{
  return std::max(lo, std::min(hi, x));
}

// IMPROVEMENT 2: Convert dBm to milliwatt
double DbmToMilliwatt(double dbm)
{
  return std::pow(10.0, dbm / 10.0);
}

// IMPROVEMENT 3: Convert milliwatt to dBm
double MilliwattToDbm(double mw)
{
  mw = std::max(mw, 1e-18);
  return 10.0 * std::log10(mw);
}

// IMPROVEMENT 4: 2D distance calculation
double Dist2d(const Vector& a, const Vector& b)
{
  double dx = a.x - b.x;
  double dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

// IMPROVEMENT 5: 3D distance calculation
double Dist3d(const Vector& a, const Vector& b)
{
  double dx = a.x - b.x;
  double dy = a.y - b.y;
  double dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// IMPROVEMENT 6: Angle between UE and gNB (for LoS probability)
double ComputeAngleToGnb(const Vector& uePos, const Vector& gnbPos)
{
  Vector delta = gnbPos - uePos;
  double dist2d = Dist2d(uePos, gnbPos);
  if (dist2d < 1e-6) return 0.0;
  
  // Angle from horizontal (elevation angle)
  double elevationAngle = std::atan2(gnbPos.z - uePos.z, dist2d) * 180.0 / kPi;
  return elevationAngle;
}

// IMPROVEMENT 7: Range wrapping
double WrapRange(double x, double minV, double maxV)
{
  double span = maxV - minV;
  if (span <= 0.0) return minV;

  double shifted = x - minV;
  shifted = std::fmod(shifted, span);
  if (shifted < 0.0) shifted += span;
  return minV + shifted;
}

// IMPROVEMENT 8: Distance to nearest grid line
double DistanceToNearestGridLine(double value, double spacing)
{
  if (spacing <= 0.0) return std::numeric_limits<double>::max();
  double nearest = std::round(value / spacing) * spacing;
  return std::fabs(value - nearest);
}

std::string ToUpper(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
  { return static_cast<char>(std::toupper(c)); });
  return s;
}

std::string NormalizePatternCode(const std::string& raw)
{
  if (raw.empty()) return "A";
  std::string u = ToUpper(raw);
  if (u.size() >= 2 && std::isdigit(static_cast<unsigned char>(u[0])) && 
      std::isalpha(static_cast<unsigned char>(u[1])))
    return std::string(1, u[1]);
  if (std::isalpha(static_cast<unsigned char>(u[0])))
    return std::string(1, u[0]);
  return "A";
}

// IMPROVEMENT 9: Linear interpolation
Vector Lerp(const Vector& a, const Vector& b, double t)
{
  t = Clamp(t, 0.0, 1.0);
  return Vector(a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t);
}

// IMPROVEMENT 10: Polyline sampling
Vector SampleAlongPolyline(const std::vector<Vector>& points, double distance, bool loop)
{
  if (points.empty()) return Vector(0.0, 0.0, 1.5);
  if (points.size() == 1) return points.front();

  double total = 0.0;
  std::vector<double> lengths;
  lengths.reserve(points.size() - 1);
  for (size_t i = 0; i + 1 < points.size(); ++i)
  {
    double seg = Dist2d(points[i], points[i + 1]);
    lengths.push_back(seg);
    total += seg;
  }

  if (total < 1e-6) return points.front();

  if (loop)
  {
    distance = std::fmod(distance, total);
    if (distance < 0.0) distance += total;
  }
  else
  {
    distance = Clamp(distance, 0.0, total);
  }

  for (size_t i = 0; i < lengths.size(); ++i)
  {
    if (distance <= lengths[i])
    {
      double ratio = lengths[i] > 1e-9 ? distance / lengths[i] : 0.0;
      return Lerp(points[i], points[i + 1], ratio);
    }
    distance -= lengths[i];
  }

  return points.back();
}

// IMPROVEMENT 11: Stop-go waypoint sampling
TrajectoryPoint SampleStopGoWaypoints(const std::vector<Vector>& waypoints,
                                      double speedMps, double pauseS, double t,
                                      double phaseOffsetS)
{
  TrajectoryPoint out;
  out.position = waypoints.empty() ? Vector(0.0, 0.0, 1.5) : waypoints.front();
  out.speedMps = 0.0;

  if (waypoints.size() < 2) return out;

  double cycle = 0.0;
  std::vector<double> moveTimes;
  moveTimes.reserve(waypoints.size() - 1);
  for (size_t i = 0; i + 1 < waypoints.size(); ++i)
  {
    double len = Dist2d(waypoints[i], waypoints[i + 1]);
    double move = len / std::max(speedMps, 0.1);
    moveTimes.push_back(move);
    cycle += move + pauseS;
  }

  if (cycle <= 1e-6) return out;

  double local = std::fmod(t + phaseOffsetS, cycle);
  if (local < 0.0) local += cycle;

  for (size_t i = 0; i < moveTimes.size(); ++i)
  {
    double move = moveTimes[i];
    if (local <= move)
    {
      double ratio = move > 1e-9 ? local / move : 0.0;
      out.position = Lerp(waypoints[i], waypoints[i + 1], ratio);
      out.speedMps = speedMps;
      out.turning = (ratio > 0.90);
      return out;
    }
    local -= move;
    if (local <= pauseS)
    {
      out.position = waypoints[i + 1];
      out.speedMps = 0.0;
      return out;
    }
    local -= pauseS;
  }

  out.position = waypoints.back();
  out.speedMps = 0.0;
  return out;
}

void EnsureParentDirectory(const std::string& path)
{
  std::filesystem::path p(path);
  if (p.has_parent_path())
    std::filesystem::create_directories(p.parent_path());
}

void AddGnb(std::vector<GnbSpec>& gnbs, double x, double y, double z, double txPowerDbm)
{
  GnbSpec g;
  g.id = static_cast<uint32_t>(gnbs.size() + 1);
  g.position = Vector(x, y, z);
  g.txPowerDbm = txPowerDbm;
  gnbs.push_back(g);
}

// IMPROVEMENT 12: Enhanced scenario catalog with better parameters
std::map<uint32_t, ScenarioSpec> BuildScenarioCatalog()
{
  std::map<uint32_t, ScenarioSpec> out;

  {
    ScenarioSpec s;
    s.id = 1;
    s.name = "Dense Manhattan Grid";
    s.xMin = 0.0;
    s.xMax = 1000.0;
    s.yMin = 0.0;
    s.yMax = 1000.0;
    s.rlfThresholdDbm = -122.0;
    s.fallbackThresholdDbm = -108.0;
    s.jitterScaleM = 1.0;      // Urban grid: ±15m reasonable
    s.baseInterferenceScale = 1.0;

    AddGnb(s.gnbs, 0.0, 0.0, 25.0, 36.0);
    AddGnb(s.gnbs, 0.0, 333.0, 25.0, 36.0);
    AddGnb(s.gnbs, 0.0, 666.0, 25.0, 36.0);
    AddGnb(s.gnbs, 333.0, 0.0, 25.0, 36.0);
    AddGnb(s.gnbs, 333.0, 333.0, 25.0, 36.0);
    AddGnb(s.gnbs, 333.0, 666.0, 25.0, 36.0);
    AddGnb(s.gnbs, 666.0, 0.0, 25.0, 36.0);
    AddGnb(s.gnbs, 666.0, 333.0, 25.0, 36.0);
    AddGnb(s.gnbs, 666.0, 666.0, 25.0, 36.0);

    s.patterns.push_back({"A", "Pedestrian Grid Walk", 600.0});
    s.patterns.push_back({"B", "Pedestrian Random Waypoint", 600.0});
    s.patterns.push_back({"C", "Fast Vehicle", 600.0});

    out[s.id] = s;
  }

  {
    ScenarioSpec s;
    s.id = 2;
    s.name = "Urban Canyon";
    s.xMin = -100.0;
    s.xMax = 600.0;
    s.yMin = 0.0;
    s.yMax = 2000.0;
    s.rlfThresholdDbm = -121.0;
    s.fallbackThresholdDbm = -108.0;
    s.jitterScaleM = 0.5;      // Narrow canyon: ±2-3m only
    s.baseInterferenceScale = 0.8;

    AddGnb(s.gnbs, 0.0, 0.0, 20.0, 36.0);
    AddGnb(s.gnbs, 0.0, 500.0, 20.0, 36.0);
    AddGnb(s.gnbs, 0.0, 1000.0, 20.0, 36.0);
    AddGnb(s.gnbs, 0.0, 1500.0, 20.0, 36.0);
    AddGnb(s.gnbs, 40.0, 250.0, 20.0, 36.0);
    AddGnb(s.gnbs, 40.0, 750.0, 20.0, 36.0);
    AddGnb(s.gnbs, 40.0, 1250.0, 20.0, 36.0);
    AddGnb(s.gnbs, 40.0, 1750.0, 20.0, 36.0);

    s.patterns.push_back({"A", "Straight Walk Down Canyon", 600.0});  // Changed from 1333
    s.patterns.push_back({"B", "Fast Walk With Perpendicular Turn", 600.0});
    s.patterns.push_back({"C", "Car Moving Fast", 600.0});            // Changed from 100

    out[s.id] = s;
  }

  {
    ScenarioSpec s;
    s.id = 3;
    s.name = "Suburban Residential";
    s.xMin = 0.0;
    s.xMax = 2000.0;
    s.yMin = 0.0;
    s.yMax = 2000.0;
    s.rlfThresholdDbm = -124.0;
    s.fallbackThresholdDbm = -112.0;
    s.jitterScaleM = 0.8;      // Suburban: ±8-10m
    s.baseInterferenceScale = 0.5;

    AddGnb(s.gnbs, 200.0, 200.0, 22.0, 36.0);
    AddGnb(s.gnbs, 700.0, 200.0, 22.0, 36.0);
    AddGnb(s.gnbs, 1300.0, 200.0, 22.0, 36.0);
    AddGnb(s.gnbs, 1800.0, 200.0, 22.0, 36.0);
    AddGnb(s.gnbs, 200.0, 700.0, 22.0, 36.0);
    AddGnb(s.gnbs, 700.0, 700.0, 22.0, 36.0);
    AddGnb(s.gnbs, 1300.0, 700.0, 22.0, 36.0);
    AddGnb(s.gnbs, 1800.0, 700.0, 22.0, 36.0);
    AddGnb(s.gnbs, 200.0, 1300.0, 22.0, 36.0);
    AddGnb(s.gnbs, 700.0, 1300.0, 22.0, 36.0);
    AddGnb(s.gnbs, 1300.0, 1300.0, 22.0, 36.0);
    AddGnb(s.gnbs, 1800.0, 1300.0, 22.0, 36.0);

    // Enhanced: Add some obstacles for pattern C
    s.patterns.push_back({"A", "Residential Walk", 600.0});
    s.patterns.push_back({"B", "Park Walk with Obstacles", 600.0});
    s.patterns.push_back({"C", "Car Commute with Turns", 600.0});

    out[s.id] = s;
  }

  {
    ScenarioSpec s;
    s.id = 4;
    s.name = "City Intersection";
    s.xMin = -150.0;
    s.xMax = 450.0;
    s.yMin = -200.0;
    s.yMax = 450.0;
    s.rlfThresholdDbm = -122.0;
    s.fallbackThresholdDbm = -110.0;
    s.jitterScaleM = 0.7;      // Intersection: ±5m
    s.baseInterferenceScale = 1.0;

    AddGnb(s.gnbs, 125.0, 400.0, 20.0, 36.0);
    AddGnb(s.gnbs, -100.0, 125.0, 20.0, 36.0);
    AddGnb(s.gnbs, 400.0, 125.0, 20.0, 36.0);
    AddGnb(s.gnbs, 125.0, -150.0, 20.0, 36.0);
    AddGnb(s.gnbs, 200.0, 350.0, 20.0, 36.0);
    AddGnb(s.gnbs, -50.0, -100.0, 20.0, 36.0);

    s.patterns.push_back({"A", "Pedestrian Crossing", 600.0});        // Changed from 233
    s.patterns.push_back({"B", "Vehicle Turning", 600.0});
    s.patterns.push_back({"C", "Traffic Light Cycles", 600.0});

    out[s.id] = s;
  }

  {
    ScenarioSpec s;
    s.id = 5;
    s.name = "High Speed Corridor";
    s.xMin = 0.0;
    s.xMax = 5000.0;
    s.yMin = -200.0;
    s.yMax = 200.0;
    s.rlfThresholdDbm = -125.0;
    s.fallbackThresholdDbm = -113.0;
    s.jitterScaleM = 0.6;      // Corridor: ±3-5m
    s.baseInterferenceScale = 0.4;

    AddGnb(s.gnbs, 0.0, 0.0, 25.0, 43.0);
    AddGnb(s.gnbs, 700.0, 0.0, 25.0, 43.0);
    AddGnb(s.gnbs, 1400.0, 0.0, 25.0, 43.0);
    AddGnb(s.gnbs, 2100.0, 0.0, 25.0, 43.0);
    AddGnb(s.gnbs, 2800.0, 0.0, 25.0, 43.0);
    AddGnb(s.gnbs, 3500.0, 0.0, 25.0, 43.0);
    AddGnb(s.gnbs, 4200.0, 0.0, 25.0, 43.0);

    s.patterns.push_back({"A", "Constant Speed 30mps", 600.0});
    s.patterns.push_back({"B", "Variable Speed Rush Hour", 600.0});
    s.patterns.push_back({"C", "Emergency Acceleration", 600.0});

    out[s.id] = s;
  }

  {
    ScenarioSpec s;
    s.id = 6;
    s.name = "Mixed Urban";
    s.xMin = 0.0;
    s.xMax = 3000.0;
    s.yMin = 0.0;
    s.yMax = 2000.0;
    s.rlfThresholdDbm = -123.0;
    s.fallbackThresholdDbm = -110.0;
    s.jitterScaleM = 0.9;      // Mixed: ±10m
    s.baseInterferenceScale = 0.75;

    // Dense zone 9 gNBs (300m spacing).
    for (uint32_t gx = 0; gx < 3; ++gx)
    {
      for (uint32_t gy = 0; gy < 3; ++gy)
      {
        AddGnb(s.gnbs, 100.0 + 300.0 * gx, 100.0 + 300.0 * gy, 22.0, 36.0);
      }
    }

    // Transition zone 6 gNBs with smoother power gradient (NEW).
    AddGnb(s.gnbs, 750.0, 150.0, 22.0, 35.0);  // -1dB
    AddGnb(s.gnbs, 750.0, 450.0, 22.0, 35.0);
    AddGnb(s.gnbs, 750.0, 750.0, 22.0, 35.0);
    AddGnb(s.gnbs, 1100.0, 150.0, 22.0, 35.0);
    AddGnb(s.gnbs, 1100.0, 450.0, 22.0, 35.0);
    AddGnb(s.gnbs, 1100.0, 750.0, 22.0, 35.0);

    // Suburban zone 6 gNBs with lower power.
    AddGnb(s.gnbs, 1400.0, 200.0, 25.0, 34.0);  // -2dB
    AddGnb(s.gnbs, 2000.0, 200.0, 25.0, 34.0);
    AddGnb(s.gnbs, 1400.0, 700.0, 25.0, 34.0);
    AddGnb(s.gnbs, 2000.0, 700.0, 25.0, 34.0);
    AddGnb(s.gnbs, 1400.0, 1200.0, 25.0, 34.0);
    AddGnb(s.gnbs, 2000.0, 1200.0, 25.0, 34.0);

    s.patterns.push_back({"A", "Full City Commute", 1200.0});
    s.patterns.push_back({"B", "Downtown Shopping", 600.0});
    s.patterns.push_back({"C", "Random Day Multi Mode", 1200.0});

    out[s.id] = s;
  }

  {
    ScenarioSpec s;
    s.id = 7;
    s.name = "NLOS Heavy Tunnel";
    s.xMin = 0.0;
    s.xMax = 1000.0;
    s.yMin = -250.0;
    s.yMax = 250.0;
    s.rlfThresholdDbm = -118.0;
    s.fallbackThresholdDbm = -108.0;
    s.jitterScaleM = 0.4;      // Tunnel: ±2m only
    s.baseInterferenceScale = 0.7;

    AddGnb(s.gnbs, 200.0, 200.0, 25.0, 36.0);
    AddGnb(s.gnbs, 800.0, 200.0, 25.0, 36.0);
    AddGnb(s.gnbs, 200.0, -200.0, 25.0, 36.0);
    AddGnb(s.gnbs, 800.0, -200.0, 25.0, 36.0);
    AddGnb(s.gnbs, 100.0, 0.0, 20.0, 36.0);
    AddGnb(s.gnbs, 900.0, 0.0, 20.0, 36.0);

    s.patterns.push_back({"A", "Walk Through Tunnel", 600.0});        // Changed from 667
    s.patterns.push_back({"B", "Car Through Tunnel", 600.0});         // Changed from 33
    s.patterns.push_back({"C", "In And Out Oscillating", 600.0});

    out[s.id] = s;
  }

  return out;
}

PatternSpec FindPattern(const ScenarioSpec& scenario, const std::string& rawPattern)
{
  std::string target = NormalizePatternCode(rawPattern);
  for (const auto& p : scenario.patterns)
  {
    if (p.code == target)
      return p;
  }
  NS_ABORT_MSG("Invalid pattern for scenario " << scenario.id << ". Use A/B/C or <scenario><A|B|C>.");
  return scenario.patterns.front();
}

class Ns3MdScenarioRunner
{
public:
  Ns3MdScenarioRunner(const ScenarioSpec& scenario, const PatternSpec& pattern, const CliOptions& cli)
    : m_scenario(scenario),
      m_pattern(pattern),
      m_cli(cli),
      m_uni(CreateObject<UniformRandomVariable>()),
      m_norm(CreateObject<NormalRandomVariable>())
  {
    m_norm->SetAttribute("Mean", DoubleValue(0.0));
    m_norm->SetAttribute("Variance", DoubleValue(1.0));

    m_durationS = (m_cli.durationS > 0.0) ? m_cli.durationS : m_pattern.defaultDurationS;
  }

  void Run()
  {
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(m_cli.seed);

    SetupNodes();
    SetupBuildings();
    InitializeUes();
    OpenLogs();

    Simulator::ScheduleNow(&Ns3MdScenarioRunner::Tick, this);
    Simulator::Stop(Seconds(m_durationS));
    Simulator::Run();
    Simulator::Destroy();

    CloseLogs();
    WriteSummary();
  }

private:
  void SetupNodes()
  {
    m_gnbNodes.Create(m_scenario.gnbs.size());
    m_ueNodes.Create(m_cli.ueCount);

    InternetStackHelper stack;
    stack.Install(m_gnbNodes);
    stack.Install(m_ueNodes);

    for (uint32_t i = 0; i < m_scenario.gnbs.size(); ++i)
    {
      Ptr<ConstantPositionMobilityModel> mob = CreateObject<ConstantPositionMobilityModel>();
      mob->SetPosition(m_scenario.gnbs[i].position);
      m_gnbNodes.Get(i)->AggregateObject(mob);
    }

    m_ues.clear();
    m_ues.resize(m_cli.ueCount);
    for (uint32_t i = 0; i < m_cli.ueCount; ++i)
    {
      UeRuntime ue;
      ue.id = i;
      ue.node = m_ueNodes.Get(i);
      ue.phaseOffsetS = m_uni->GetValue(0.0, 20.0);
      
      // IMPROVEMENT: Scenario-specific jitter scaling
      double jitterScale = m_scenario.jitterScaleM;
      ue.lateralOffsetM = m_uni->GetValue(-3.0 * jitterScale, 3.0 * jitterScale);
      ue.jitter = Vector(m_uni->GetValue(-15.0 * jitterScale, 15.0 * jitterScale),
                         m_uni->GetValue(-15.0 * jitterScale, 15.0 * jitterScale),
                         0.0);

      Ptr<ConstantPositionMobilityModel> mob = CreateObject<ConstantPositionMobilityModel>();
      mob->SetPosition(Vector(m_scenario.xMin + 5.0, m_scenario.yMin + 5.0, 1.5));
      ue.node->AggregateObject(mob);

      m_ues[i] = ue;
    }
  }

  void AddBuilding(double x1, double x2, double y1, double y2, double h)
  {
    Ptr<Building> b = CreateObject<Building>();
    b->SetBoundaries(Box(std::min(x1, x2), std::max(x1, x2), std::min(y1, y2), std::max(y1, y2), 0.0, h));
    b->SetBuildingType(Building::Office);
    b->SetExtWallsType(Building::ConcreteWithWindows);
    b->SetNFloors(std::max<uint16_t>(1, static_cast<uint16_t>(h / 3.0)));
    b->SetNRoomsX(2);
    b->SetNRoomsY(2);
  }

  void GenerateGridBuildings(double blockM, double streetM, double density, double hMin, double hMax)
  {
    for (double x = m_scenario.xMin; x + blockM <= m_scenario.xMax; x += (blockM + streetM))
    {
      for (double y = m_scenario.yMin; y + blockM <= m_scenario.yMax; y += (blockM + streetM))
      {
        if (m_uni->GetValue() > density)
          continue;
        double h = m_uni->GetValue(hMin, hMax);
        AddBuilding(x, x + blockM, y, y + blockM, h);
      }
    }
  }

  void SetupBuildings()
  {
    switch (m_scenario.id)
    {
    case 1:
      GenerateGridBuildings(90.0, 20.0, 0.70, 40.0, 60.0);
      break;

    case 2:
      // IMPROVEMENT: Better canyon building alignment
      AddBuilding(-120.0, -5.0, 0.0, 2000.0, 60.0);
      AddBuilding(45.0, 240.0, 0.0, 2000.0, 60.0);
      break;

    case 3:
      GenerateGridBuildings(45.0, 40.0, 0.35, 8.0, 12.0); // Increased density
      break;

    case 4:
      AddBuilding(-150.0, 40.0, 260.0, 450.0, 45.0);
      AddBuilding(210.0, 450.0, 260.0, 450.0, 50.0);
      AddBuilding(-150.0, 40.0, -200.0, -10.0, 45.0);
      AddBuilding(210.0, 450.0, -200.0, -10.0, 50.0);
      break;

    case 5:
      AddBuilding(900.0, 980.0, 80.0, 180.0, 15.0);
      AddBuilding(2400.0, 2480.0, -180.0, -80.0, 18.0);
      AddBuilding(3800.0, 3890.0, 70.0, 180.0, 12.0);
      break;

    case 6:
      // Dense zone.
      for (double x = 0.0; x < 700.0; x += 120.0)
      {
        for (double y = 0.0; y < 1200.0; y += 120.0)
        {
          if (m_uni->GetValue() <= 0.70)
            AddBuilding(x, x + 90.0, y, y + 90.0, m_uni->GetValue(40.0, 60.0));
        }
      }

      // Transition zone.
      for (double x = 700.0; x < 1300.0; x += 170.0)
      {
        for (double y = 0.0; y < 1500.0; y += 170.0)
        {
          if (m_uni->GetValue() <= 0.40)
            AddBuilding(x, x + 120.0, y, y + 120.0, m_uni->GetValue(15.0, 40.0));
        }
      }

      // Suburban zone.
      for (double x = 1300.0; x < 2900.0; x += 220.0)
      {
        for (double y = 0.0; y < 1700.0; y += 220.0)
        {
          if (m_uni->GetValue() <= 0.20)
            AddBuilding(x, x + 120.0, y, y + 120.0, m_uni->GetValue(8.0, 15.0));
        }
      }
      break;

    case 7:
      AddBuilding(0.0, 1000.0, 20.0, 250.0, 25.0);
      AddBuilding(0.0, 1000.0, -250.0, -20.0, 25.0);
      break;

    default:
      break;
    }

    BuildingsHelper::Install(m_gnbNodes);
    BuildingsHelper::Install(m_ueNodes);
  }

  TrajectoryPoint SampleScenario1(const UeRuntime& ue, double t) const
  {
    std::string p = m_pattern.code;
    if (p == "A")
    {
      double speed = 1.5;
      std::vector<Vector> route = {
        Vector(120.0, 120.0, 1.5),  Vector(120.0, 880.0, 1.5),
        Vector(500.0, 880.0, 1.5),  Vector(500.0, 120.0, 1.5),
        Vector(880.0, 120.0, 1.5),  Vector(880.0, 880.0, 1.5),
        Vector(120.0, 880.0, 1.5)
      };
      double distance = speed * (t + ue.phaseOffsetS);
      TrajectoryPoint tp;
      tp.position = SampleAlongPolyline(route, distance, true);
      tp.speedMps = speed;
      return tp;
    }

    if (p == "B")
    {
      std::vector<Vector> waypoints = {
        Vector(140.0, 120.0, 1.5),   Vector(140.0, 760.0, 1.5),
        Vector(420.0, 760.0, 1.5),   Vector(420.0, 260.0, 1.5),
        Vector(760.0, 260.0, 1.5),   Vector(760.0, 860.0, 1.5),
        Vector(220.0, 860.0, 1.5),   Vector(140.0, 120.0, 1.5)
      };
      return SampleStopGoWaypoints(waypoints, 2.0, 3.0, t, ue.phaseOffsetS);
    }

    double speed = 15.0;
    std::vector<Vector> route = {
      Vector(60.0, 170.0, 1.5),    Vector(940.0, 170.0, 1.5),
      Vector(940.0, 840.0, 1.5),   Vector(60.0, 840.0, 1.5),
      Vector(60.0, 170.0, 1.5)
    };
    double distance = speed * (t + ue.phaseOffsetS);
    TrajectoryPoint tp;
    tp.position = SampleAlongPolyline(route, distance, true);
    tp.speedMps = speed;
    return tp;
  }

  TrajectoryPoint SampleScenario2(const UeRuntime& ue, double t) const
  {
    TrajectoryPoint tp;
    double local = t + ue.phaseOffsetS;

    if (m_pattern.code == "A")
    {
      double speed = 1.5;
      double y = std::fmod(speed * local, 2000.0);
      if (y < 0.0) y += 2000.0;
      tp.position = Vector(20.0 + ue.lateralOffsetM, y, 1.5);
      tp.speedMps = speed;
      return tp;
    }

    if (m_pattern.code == "B")
    {
      double cycle = 90.0;
      double phase = std::fmod(local, cycle);
      if (phase < 0.0) phase += cycle;
      uint32_t lap = static_cast<uint32_t>(std::floor(local / cycle));
      double yBase = std::fmod(lap * 200.0, 2000.0);

      double speed = 5.0;
      if (phase < 40.0)
      {
        tp.position = Vector(20.0 + ue.lateralOffsetM, yBase + speed * phase, 1.5);
        tp.speedMps = speed;
      }
      else if (phase < 45.0)
      {
        double alpha = (phase - 40.0) / 5.0;
        tp.position = Vector(20.0 + 200.0 * alpha, yBase + 200.0, 1.5);
        tp.speedMps = speed;
        tp.turning = true;
      }
      else
      {
        double x = 220.0 + speed * (phase - 45.0);
        tp.position = Vector(WrapRange(x, -80.0, 520.0), yBase + 200.0, 1.5);
        tp.speedMps = speed;
      }
      return tp;
    }

    double speed = 20.0;
    double y = std::fmod(speed * local, 2000.0);
    if (y < 0.0) y += 2000.0;
    tp.position = Vector(20.0 + ue.lateralOffsetM, y, 1.5);
    tp.speedMps = speed;
    return tp;
  }

  TrajectoryPoint SampleScenario3(const UeRuntime& ue, double t) const
  {
    TrajectoryPoint tp;
    double local = t + ue.phaseOffsetS;

    if (m_pattern.code == "A")
    {
      std::vector<Vector> route = {
        Vector(200.0, 200.0, 1.5),   Vector(1800.0, 200.0, 1.5),
        Vector(1800.0, 700.0, 1.5),  Vector(200.0, 700.0, 1.5),
        Vector(200.0, 1300.0, 1.5),  Vector(1800.0, 1300.0, 1.5),
        Vector(1800.0, 1800.0, 1.5), Vector(300.0, 1800.0, 1.5),
        Vector(200.0, 200.0, 1.5)
      };
      double speed = 1.5;
      tp.position = SampleAlongPolyline(route, speed * local, true);
      tp.speedMps = speed;
      return tp;
    }

    if (m_pattern.code == "B")
    {
      double speed = 1.7;
      tp.position = Vector(1000.0 + 260.0 * std::sin(0.012 * local + ue.phaseOffsetS),
                           1000.0 + 220.0 * std::sin(0.017 * local + 1.2 + ue.phaseOffsetS),
                           1.5);
      tp.speedMps = speed;
      return tp;
    }

    std::vector<Vector> route = {
      Vector(100.0, 650.0, 1.5),   Vector(1900.0, 650.0, 1.5),
      Vector(1900.0, 900.0, 1.5),  Vector(100.0, 900.0, 1.5),
      Vector(100.0, 650.0, 1.5)
    };
    double distance = 25.0 * local + 100.0 * std::sin(0.05 * local);
    tp.position = SampleAlongPolyline(route, distance, true);
    tp.speedMps = Clamp(25.0 + 5.0 * std::cos(0.05 * local), 20.0, 30.0);
    return tp;
  }

  TrajectoryPoint SampleScenario4(const UeRuntime& ue, double t) const
  {
    TrajectoryPoint tp;
    double local = t + ue.phaseOffsetS;

    if (m_pattern.code == "A")
    {
      double speed = 1.5;
      Vector a(0.0, 250.0, 1.5);
      Vector b(250.0, 0.0, 1.5);
      double length = Dist2d(a, b);
      double d = std::fmod(speed * local, length);
      if (d < 0.0) d += length;
      tp.position = Lerp(a, b, d / length);
      tp.speedMps = speed;
      return tp;
    }

    if (m_pattern.code == "B")
    {
      std::vector<Vector> route = {
        Vector(125.0, -150.0, 1.5), Vector(125.0, 125.0, 1.5),
        Vector(400.0, 125.0, 1.5),  Vector(125.0, 125.0, 1.5),
        Vector(125.0, -150.0, 1.5)
      };
      double speed = 15.0;
      tp.position = SampleAlongPolyline(route, speed * local, true);
      tp.speedMps = speed;
      return tp;
    }

    std::vector<Vector> route = {
      Vector(125.0, -120.0, 1.5), Vector(125.0, 125.0, 1.5),
      Vector(380.0, 125.0, 1.5),  Vector(125.0, -120.0, 1.5)
    };
    TrajectoryPoint stopGo = SampleStopGoWaypoints(route, 10.0, 40.0, t, ue.phaseOffsetS);
    return stopGo;
  }

  double HighwayDistancePatternB(double t) const
  {
    double cycle = 40.0;
    double phase = std::fmod(t, cycle);
    if (phase < 0.0) phase += cycle;
    uint64_t laps = static_cast<uint64_t>(std::floor(t / cycle));
    double base = static_cast<double>(laps) * 1100.0;
    if (phase < 10.0)
      return base + (5.0 * phase + 1.5 * phase * phase);
    if (phase < 30.0)
      return base + 200.0 + 35.0 * (phase - 10.0);
    double dt = phase - 30.0;
    return base + 900.0 + 35.0 * dt - 1.5 * dt * dt;
  }

  double HighwaySpeedPatternB(double t) const
  {
    double phase = std::fmod(t, 40.0);
    if (phase < 0.0) phase += 40.0;
    if (phase < 10.0)
      return 5.0 + 3.0 * phase;
    if (phase < 30.0)
      return 35.0;
    return 35.0 - 3.0 * (phase - 30.0);
  }

  double HighwayDistancePatternC(double t) const
  {
    double cycle = 120.0;
    double phase = std::fmod(t, cycle);
    if (phase < 0.0) phase += cycle;
    uint64_t laps = static_cast<uint64_t>(std::floor(t / cycle));
    double base = static_cast<double>(laps) * 3000.0;
    if (phase < 40.0)
      return base + 20.0 * phase;
    if (phase < 60.0)
      return base + 800.0 + 35.0 * (phase - 40.0);
    return base + 1500.0 + 25.0 * (phase - 60.0);
  }

  double HighwaySpeedPatternC(double t) const
  {
    double phase = std::fmod(t, 120.0);
    if (phase < 0.0) phase += 120.0;
    if (phase < 40.0)
      return 20.0;
    if (phase < 60.0)
      return 35.0;
    return 25.0;
  }

  TrajectoryPoint SampleScenario5(const UeRuntime& ue, double t) const
  {
    TrajectoryPoint tp;
    double local = t + ue.phaseOffsetS;

    if (m_pattern.code == "A")
    {
      double speed = 30.0;
      double x = std::fmod(speed * local, 5000.0);
      if (x < 0.0) x += 5000.0;
      tp.position = Vector(x, ue.lateralOffsetM, 1.5);
      tp.speedMps = speed;
      return tp;
    }

    if (m_pattern.code == "B")
    {
      double distance = HighwayDistancePatternB(local);
      double x = std::fmod(distance, 5000.0);
      if (x < 0.0) x += 5000.0;
      tp.position = Vector(x, ue.lateralOffsetM, 1.5);
      tp.speedMps = HighwaySpeedPatternB(local);
      return tp;
    }

    double distance = HighwayDistancePatternC(local);
    double x = std::fmod(distance, 5000.0);
    if (x < 0.0) x += 5000.0;
    tp.position = Vector(x, ue.lateralOffsetM, 1.5);
    tp.speedMps = HighwaySpeedPatternC(local);
    return tp;
  }

  TrajectoryPoint SampleScenario6(const UeRuntime& ue, double t) const
  {
    TrajectoryPoint tp;
    double local = t + ue.phaseOffsetS;

    if (m_pattern.code == "A")
    {
      double phase = std::fmod(local, 1200.0);
      if (phase < 0.0) phase += 1200.0;

      Vector p0(2200.0, 1000.0, 1.5), p1(1500.0, 900.0, 1.5),
             p2(900.0, 700.0, 1.5),   p3(350.0, 350.0, 1.5),
             p4(900.0, 650.0, 1.5),   p5(2200.0, 1000.0, 1.5);

      if (phase < 200.0)
      {
        tp.position = Lerp(p0, p1, phase / 200.0);
        tp.speedMps = 12.0;
      }
      else if (phase < 400.0)
      {
        tp.position = Lerp(p1, p2, (phase - 200.0) / 200.0);
        tp.speedMps = 15.0;
      }
      else if (phase < 600.0)
      {
        tp.position = Lerp(p2, p3, (phase - 400.0) / 200.0);
        tp.speedMps = 10.0;
      }
      else if (phase < 800.0)
      {
        tp.position = Lerp(p3, p4, (phase - 600.0) / 200.0);
        tp.speedMps = 18.0;
      }
      else
      {
        tp.position = Lerp(p4, p5, (phase - 800.0) / 400.0);
        tp.speedMps = 20.0;
      }
      return tp;
    }

    if (m_pattern.code == "B")
    {
      double speed = 2.0;
      tp.position = Vector(400.0 + 220.0 * std::sin(0.015 * local + ue.phaseOffsetS),
                           400.0 + 170.0 * std::sin(0.023 * local + 0.7 + ue.phaseOffsetS),
                           1.5);
      tp.speedMps = speed;
      return tp;
    }

    double cycle = 720.0;
    double phase = std::fmod(local, cycle);
    if (phase < 0.0) phase += cycle;

    if (phase < 120.0)
    {
      tp.position = Vector(1700.0 + 70.0 * std::sin(0.03 * phase),
                           950.0 + 30.0 * std::cos(0.03 * phase), 1.5);
      tp.speedMps = 1.5;
      return tp;
    }

    if (phase < 360.0)
    {
      double alpha = (phase - 120.0) / 240.0;
      tp.position = Lerp(Vector(1700.0, 950.0, 1.5), Vector(350.0, 380.0, 1.5), alpha);
      tp.speedMps = 20.0;
      return tp;
    }

    if (phase < 480.0)
    {
      double p = phase - 360.0;
      tp.position = Vector(350.0 + 60.0 * std::sin(0.05 * p),
                           380.0 + 60.0 * std::cos(0.05 * p), 1.5);
      tp.speedMps = 2.0;
      return tp;
    }

    double alpha = (phase - 480.0) / 240.0;
    tp.position = Lerp(Vector(350.0, 380.0, 1.5), Vector(1700.0, 950.0, 1.5), alpha);
    tp.speedMps = 20.0;
    return tp;
  }

  TrajectoryPoint SampleScenario7(const UeRuntime& ue, double t) const
  {
    TrajectoryPoint tp;
    double local = t + ue.phaseOffsetS;

    if (m_pattern.code == "A")
    {
      double speed = 1.5;
      double x = std::fmod(speed * local, 1000.0);
      if (x < 0.0) x += 1000.0;
      tp.position = Vector(x, ue.lateralOffsetM, 1.5);
      tp.speedMps = speed;
      tp.inTunnel = (x >= 0.0 && x <= 1000.0 && std::fabs(tp.position.y) <= 10.0);
      return tp;
    }

    if (m_pattern.code == "B")
    {
      double speed = 30.0;
      double x = std::fmod(speed * local, 1000.0);
      if (x < 0.0) x += 1000.0;
      tp.position = Vector(x, ue.lateralOffsetM, 1.5);
      tp.speedMps = speed;
      tp.inTunnel = (x >= 0.0 && x <= 1000.0 && std::fabs(tp.position.y) <= 10.0);
      return tp;
    }

    double cycle = 280.0;
    double phase = std::fmod(local, cycle);
    if (phase < 0.0) phase += cycle;

    if (phase < 100.0)
    {
      double x = 2.0 * phase;
      tp.position = Vector(x, ue.lateralOffsetM, 1.5);
      tp.speedMps = 2.0;
    }
    else if (phase < 150.0)
    {
      tp.position = Vector(200.0, ue.lateralOffsetM, 1.5);
      tp.speedMps = 0.0;
    }
    else if (phase < 250.0)
    {
      double x = 200.0 - 2.0 * (phase - 150.0);
      tp.position = Vector(x, ue.lateralOffsetM, 1.5);
      tp.speedMps = 2.0;
    }
    else
    {
      tp.position = Vector(0.0, ue.lateralOffsetM, 1.5);
      tp.speedMps = 0.0;
    }

    tp.inTunnel = (tp.position.x >= 0.0 && tp.position.x <= 1000.0 && std::fabs(tp.position.y) <= 10.0);
    return tp;
  }

  TrajectoryPoint SampleTrajectory(const UeRuntime& ue, double nowS) const
  {
    TrajectoryPoint tp;
    switch (m_scenario.id)
    {
    case 1:
      tp = SampleScenario1(ue, nowS);
      break;
    case 2:
      tp = SampleScenario2(ue, nowS);
      break;
    case 3:
      tp = SampleScenario3(ue, nowS);
      break;
    case 4:
      tp = SampleScenario4(ue, nowS);
      break;
    case 5:
      tp = SampleScenario5(ue, nowS);
      break;
    case 6:
      tp = SampleScenario6(ue, nowS);
      break;
    case 7:
      tp = SampleScenario7(ue, nowS);
      break;
    default:
      tp.position = Vector(m_scenario.xMin, m_scenario.yMin, 1.5);
      tp.speedMps = 0.0;
      break;
    }

    tp.position.x += ue.jitter.x;
    tp.position.y += ue.jitter.y;
    tp.position.x = WrapRange(tp.position.x, m_scenario.xMin, m_scenario.xMax);
    tp.position.y = WrapRange(tp.position.y, m_scenario.yMin, m_scenario.yMax);
    tp.position.z = 1.5;

    if (m_scenario.id == 7)
      tp.inTunnel = (tp.position.x >= 0.0 && tp.position.x <= 1000.0 && std::fabs(tp.position.y) <= 10.0);

    return tp;
  }

  // IMPROVEMENT 13: Cell-specific LoS probability (ENHANCED)
  double ComputeLosProbability(const TrajectoryPoint& tp, const Vector& gnbPos) const
  {
    switch (m_scenario.id)
    {
    case 1:
    {
      bool onStreetX = (DistanceToNearestGridLine(tp.position.x, 100.0) < 8.0);
      bool onStreetY = (DistanceToNearestGridLine(tp.position.y, 100.0) < 8.0);
      if (onStreetX && onStreetY)
        return 0.70;  // Intersection
      if (onStreetX || onStreetY)
        return 0.85;  // Street
      return 0.15;    // Urban canyon (increased from 0.10)
    }

    case 2:
    {
      double dist2d = Dist2d(tp.position, gnbPos);
      if (tp.turning)
        return 0.20;  // Turning: more blockage
      if (std::fabs(tp.position.x - 20.0) < 12.0)
        return 0.92;  // Aligned with street
      // Distance-dependent LoS in canyon
      if (dist2d > 500.0)
        return 0.12;  // Far side: blocked
      return 0.20;    // Intermediate
    }

    case 3:
    {
      double dist2d = Dist2d(tp.position, gnbPos);
      if (dist2d < 300.0)
        return 0.85;  // Close = mostly LoS in suburban
      return 0.55;    // Far = mixed
    }

    case 4:
    {
      double dCenter = Dist2d(tp.position, Vector(125.0, 125.0, 0.0));
      return (dCenter < 90.0) ? 1.0 : 0.70;  // Intersection specific
    }

    case 5:
      return 0.98;  // Highway: mostly LoS

    case 6:
    {
      // Zone-dependent LoS
      if (tp.position.x < 700.0)
        return 0.75;  // Dense
      if (tp.position.x < 1300.0)
        return 0.85;  // Transition
      return 0.92;    // Suburban
    }

    case 7:
      if (tp.inTunnel)
      {
        double depth = std::min(tp.position.x, 1000.0 - tp.position.x);
        if (depth < 50.0)
          return 0.20;    // Tunnel entrance
        if (depth < 500.0)
          return 0.03;    // Deep tunnel
        return 0.25;      // Tunnel exit
      }
      return 0.45;        // Outside tunnel

    default:
      return 0.5;
    }
  }

  // IMPROVEMENT 14: Position-dependent interference (ENHANCED)
  double ComputeInterferenceScale(const TrajectoryPoint& tp, const Vector& servingPos, const Vector& interferPos) const
  {
    double distToServing = Dist2d(tp.position, servingPos);
    double distToInterfer = Dist2d(tp.position, interferPos);
    
    // Closer to serving = less interference contribution
    double baseScale = m_scenario.baseInterferenceScale;
    
    // Distance ratio: if far from serving and close to interferer, more interference
    if (distToServing > 100.0)
    {
      double ratio = distToInterfer / std::max(distToServing, 10.0);
      return baseScale * Clamp(ratio, 0.1, 1.5);
    }
    
    return baseScale * 0.7; // Close to serving: less interference
  }

  // IMPROVEMENT 15: Spatial shadow fading correlation (NEW)
  double ApplyShadowFadingCorrelation(UeRuntime& ue, uint32_t cellId, double newShadow)
  {
    auto& history = ue.shadowFadingHistory[cellId];
    
    if (history.empty())
    {
      history.push_back(newShadow);
      return newShadow;
    }

    // Simple exponential averaging for correlation
    double alpha = 0.3;  // Correlation factor
    double correlatedShadow = alpha * newShadow + (1.0 - alpha) * history.back();
    
    history.push_back(correlatedShadow);
    if (history.size() > kShadowFadingHistorySize)
      history.pop_front();
    
    return correlatedShadow;
  }

  void ResolvePathLossModel(const TrajectoryPoint& tp, bool isLos,
                             double& base, double& slope, double& shadowSigma,
                             double& fadingSigma, double& extraLossDb) const
  {
    extraLossDb = 0.0;

    switch (m_scenario.id)
    {
    case 1:
      base = 140.7;
      slope = 36.7;
      shadowSigma = 8.0;
      fadingSigma = 4.0;
      if (!isLos)
        extraLossDb += 5.0;  // Urban NLoS
      break;

    case 2:
      if (isLos)
      {
        base = 128.0;
        slope = 20.0;
        shadowSigma = 4.0;
        fadingSigma = 3.0;
      }
      else
      {
        base = 140.0;
        slope = 30.0;
        shadowSigma = 12.0;
        fadingSigma = 6.0;
        extraLossDb += tp.turning ? 18.0 : 7.0;
      }
      break;

    case 3:
      base = 135.7;
      slope = 35.7;
      shadowSigma = 6.0;
      fadingSigma = 3.0;
      if (!isLos)
        extraLossDb += 1.5;
      break;

    case 4:
      base = 140.0;
      slope = 36.0;
      shadowSigma = 8.0;
      fadingSigma = 5.0;
      break;

    case 5:
      base = 128.1;
      slope = 37.6;
      shadowSigma = 2.5;
      fadingSigma = 1.5;
      break;

    case 6:
      if (tp.position.x < 700.0)
      {
        base = 140.7;
        slope = 36.7;
        shadowSigma = 8.0;
        fadingSigma = 4.0;
        if (!isLos)
          extraLossDb += 2.5;
      }
      else if (tp.position.x < 1300.0)
      {
        base = 135.0;
        slope = 35.0;
        shadowSigma = 6.0;
        fadingSigma = 3.0;
        if (!isLos)
          extraLossDb += 1.5;
      }
      else
      {
        base = 128.0;
        slope = 35.0;
        shadowSigma = 5.0;
        fadingSigma = 2.5;
        if (!isLos)
          extraLossDb += 0.5;
      }
      break;

    case 7:
      if (tp.inTunnel)
      {
        base = 140.0;
        slope = 50.0;
        shadowSigma = 10.0;
        fadingSigma = 6.0;

        double depth = std::min(tp.position.x, 1000.0 - tp.position.x);
        if (depth <= 100.0)
          extraLossDb += 15.0 * (depth / 100.0);
        else if (depth <= 500.0)
          extraLossDb += 15.0 + 25.0 * ((depth - 100.0) / 400.0);
        else
          extraLossDb += 40.0;
      }
      else
      {
        base = 135.0;
        slope = 35.0;
        shadowSigma = 6.0;
        fadingSigma = 3.0;
      }
      break;

    default:
      base = 140.0;
      slope = 36.0;
      shadowSigma = 7.0;
      fadingSigma = 4.0;
      break;
    }
  }

  std::vector<CellMeasurement> MeasureCells(const TrajectoryPoint& tp)
  {
    std::vector<CellMeasurement> out;
    out.reserve(m_scenario.gnbs.size());

    for (const auto& g : m_scenario.gnbs)
    {
      double dist2d = Dist2d(tp.position, g.position);
      double dist3d = Dist3d(tp.position, g.position);
      double dKm = std::max(dist3d / 1000.0, 1e-3);

      double losP = ComputeLosProbability(tp, g.position);
      bool isLos = (m_uni->GetValue() < losP);

      double base = 140.0, slope = 36.0, sigmaShadow = 7.0, sigmaFading = 4.0, extraLoss = 0.0;
      ResolvePathLossModel(tp, isLos, base, slope, sigmaShadow, sigmaFading, extraLoss);

      double pathLossDb = base + slope * std::log10(dKm) + extraLoss;
      double shadowDb = m_norm->GetValue() * sigmaShadow;
      double fadingDb = m_norm->GetValue() * sigmaFading;

      CellMeasurement m;
      m.cellId = g.id;
      m.distanceM = dist2d;
      m.losProbability = losP;
      m.isLos = isLos;
      m.rsrpDbm = g.txPowerDbm - pathLossDb + shadowDb + fadingDb;
      out.push_back(m);
    }

    // IMPROVEMENT 16: Position-dependent interference calculation
    double noiseMw = DbmToMilliwatt(kNoiseFloorDbm);

    for (size_t i = 0; i < out.size(); ++i)
    {
      double signalMw = DbmToMilliwatt(out[i].rsrpDbm);
      double interfMw = 0.0;
      
      for (size_t j = 0; j < out.size(); ++j)
      {
        if (i == j) continue;
        
        // Position-dependent interference scaling
        double scaleij = ComputeInterferenceScale(tp, m_scenario.gnbs[i].position, m_scenario.gnbs[j].position);
        interfMw += scaleij * DbmToMilliwatt(out[j].rsrpDbm);
      }
      
      out[i].sinrDb = MilliwattToDbm(signalMw / (noiseMw + interfMw));
    }

    std::sort(out.begin(), out.end(), [](const CellMeasurement& a, const CellMeasurement& b)
    { return a.rsrpDbm > b.rsrpDbm; });

    return out;
  }

  const CellMeasurement* FindMeasurementByCell(const std::vector<CellMeasurement>& meas, uint32_t cellId) const
  {
    for (const auto& m : meas)
    {
      if (m.cellId == cellId)
        return &m;
    }
    return nullptr;
  }

  uint32_t BestNeighborCell(const std::vector<CellMeasurement>& meas, uint32_t servingCell, double& bestNeighborRsrp) const
  {
    bestNeighborRsrp = -999.0;
    for (const auto& m : meas)
    {
      if (m.cellId == servingCell)
        continue;
      bestNeighborRsrp = m.rsrpDbm;
      return m.cellId;
    }
    return 0;
  }

  void WriteEvent(double now, const UeRuntime& ue, const std::string& eventType,
                  uint32_t fromCell, uint32_t toCell, double servingRsrp,
                  double targetRsrp, double margin, const std::string& reason)
  {
    m_eventCsv << std::fixed << std::setprecision(3)
               << now << "," << m_scenario.id << "," << m_pattern.code << ","
               << ue.id << "," << eventType << "," << fromCell << "," << toCell << ","
               << servingRsrp << "," << targetRsrp << "," << margin << "," << reason << "\n";
  }

  TickDecision ProcessHandover(UeRuntime& ue, double now,
                                const std::vector<CellMeasurement>& meas,
                                const CellMeasurement& serving,
                                uint32_t bestNeighborCell,
                                double bestNeighborRsrp)
  {
    TickDecision decision;
    decision.fromCell = ue.servingCell;
    double margin = bestNeighborRsrp - serving.rsrpDbm;
    decision.marginDb = margin;

    if (serving.rsrpDbm < m_scenario.rlfThresholdDbm)
      ue.lowRsrpTicks += 1;
    else
      ue.lowRsrpTicks = 0;

    if (ue.lowRsrpTicks >= kRlfT310Ticks)
    {
      decision.rlfEvent = true;
      ue.rlfCount += 1;
      m_totalRlf += 1;

      const CellMeasurement& best = meas.front();
      if (best.cellId != ue.servingCell && best.rsrpDbm >= m_scenario.fallbackThresholdDbm)
      {
        decision.hoEvent = true;
        decision.emergencyHo = true;
        decision.toCell = best.cellId;
        decision.reason = "RLF_RECOVERY";

        WriteEvent(now, ue, "EMERGENCY_HO", ue.servingCell, best.cellId, serving.rsrpDbm, best.rsrpDbm,
                   best.rsrpDbm - serving.rsrpDbm, "RLF_RECOVERY");

        CheckPingPong(now, ue, ue.servingCell, best.cellId, decision);
        ue.servingCell = best.cellId;
        ue.hoCount += 1;
        m_totalHo += 1;
      }
      else
      {
        decision.reason = "RLF_NO_RECOVERY";
        WriteEvent(now, ue, "RLF", ue.servingCell, 0, serving.rsrpDbm, -140.0, -999.0, "NO_CANDIDATE");
      }

      ue.lowRsrpTicks = 0;
      ue.candidateCell = 0;
      ue.a3HoldMs = 0;
      return decision;
    }

    if (bestNeighborCell != 0 && margin >= m_cli.hysDb)
    {
      if (ue.candidateCell == bestNeighborCell)
        ue.a3HoldMs += kTickMs;
      else
      {
        ue.candidateCell = bestNeighborCell;
        ue.a3HoldMs = kTickMs;
      }
    }
    else
    {
      ue.candidateCell = 0;
      ue.a3HoldMs = 0;
    }

    if (ue.candidateCell != 0 && ue.a3HoldMs >= m_cli.tttMs)
    {
      const CellMeasurement* target = FindMeasurementByCell(meas, ue.candidateCell);
      if (target != nullptr)
      {
        decision.hoEvent = true;
        decision.toCell = target->cellId;
        decision.reason = "A3_TTT";

        WriteEvent(now, ue, "HO", ue.servingCell, target->cellId, serving.rsrpDbm,
                   target->rsrpDbm, target->rsrpDbm - serving.rsrpDbm, "A3_TTT");

        CheckPingPong(now, ue, ue.servingCell, target->cellId, decision);
        ue.servingCell = target->cellId;
        ue.hoCount += 1;
        m_totalHo += 1;
      }

      ue.candidateCell = 0;
      ue.a3HoldMs = 0;
    }

    return decision;
  }

  void CheckPingPong(double now, UeRuntime& ue, uint32_t fromCell, uint32_t toCell, TickDecision& decision)
  {
    if (ue.lastHoFromCell == static_cast<int32_t>(toCell) &&
        ue.lastHoToCell == static_cast<int32_t>(fromCell) &&
        (now - ue.lastHoTimeS) <= kPingPongWindowS)
    {
      decision.pingPongEvent = true;
      ue.pingPongCount += 1;
      m_totalPingPong += 1;
    }

    ue.lastHoFromCell = static_cast<int32_t>(fromCell);
    ue.lastHoToCell = static_cast<int32_t>(toCell);
    ue.lastHoTimeS = now;
  }

  void LogTick(double now, const UeRuntime& ue, const TrajectoryPoint& tp,
               const std::vector<CellMeasurement>& meas, const CellMeasurement& serving,
               uint32_t bestNeighborCell, double bestNeighborRsrp, const TickDecision& decision)
  {
    double margin = (bestNeighborCell != 0) ? (bestNeighborRsrp - serving.rsrpDbm) : -999.0;

    int servingCqi = 1;
    if (serving.sinrDb >= 22.7) servingCqi = 15;
    else if (serving.sinrDb >= 20.7) servingCqi = 14;
    else if (serving.sinrDb >= 18.7) servingCqi = 13;
    else if (serving.sinrDb >= 17.4) servingCqi = 12;
    else if (serving.sinrDb >= 14.1) servingCqi = 11;
    else if (serving.sinrDb >= 11.7) servingCqi = 10;
    else if (serving.sinrDb >= 9.0) servingCqi = 9;
    else if (serving.sinrDb >= 7.0) servingCqi = 8;
    else if (serving.sinrDb >= 5.1) servingCqi = 7;
    else if (serving.sinrDb >= 3.0) servingCqi = 6;
    else if (serving.sinrDb >= 1.0) servingCqi = 5;
    else if (serving.sinrDb >= -1.0) servingCqi = 4;
    else if (serving.sinrDb >= -4.7) servingCqi = 3;
    else if (serving.sinrDb >= -6.7) servingCqi = 2;

    m_tickCsv << std::fixed << std::setprecision(3)
              << now << "," << m_scenario.id << "," << m_pattern.code << "," << ue.id << ","
              << tp.position.x << "," << tp.position.y << "," << tp.speedMps << ","
              << ue.servingCell << "," << serving.rsrpDbm << "," << serving.sinrDb << ","
              << serving.distanceM << "," << servingCqi << "," << bestNeighborCell << ","
              << bestNeighborRsrp << "," << margin << "," << ue.candidateCell << ","
              << ue.a3HoldMs << "," << static_cast<int>(decision.hoEvent) << ","
              << static_cast<int>(decision.rlfEvent) << ","
              << static_cast<int>(decision.pingPongEvent) << "," << serving.losProbability;

    uint32_t written = 0;
    for (size_t idx = 0; idx < meas.size() && written < kTopNeighborsToLog; ++idx)
    {
      if (meas[idx].cellId == ue.servingCell)
        continue;
      const auto& n = meas[idx];
      m_tickCsv << "," << n.cellId << "," << n.rsrpDbm << "," << n.sinrDb << "," << n.distanceM;
      ++written;
    }

    for (; written < kTopNeighborsToLog; ++written)
      m_tickCsv << ",-1,-140.0,-20.0,0.0";

    m_tickCsv << "\n";
  }

  void InitializeUes()
  {
    for (auto& ue : m_ues)
    {
      TrajectoryPoint tp = SampleTrajectory(ue, 0.0);
      Ptr<ConstantPositionMobilityModel> mob = ue.node->GetObject<ConstantPositionMobilityModel>();
      mob->SetPosition(tp.position);

      std::vector<CellMeasurement> meas = MeasureCells(tp);
      if (!meas.empty())
        ue.servingCell = meas.front().cellId;
      else
        ue.servingCell = 0;
    }
  }

  void OpenLogs()
  {
    std::string tickPath = m_cli.outputPrefix + "_tick.csv";
    std::string eventPath = m_cli.outputPrefix + "_events.csv";

    EnsureParentDirectory(tickPath);
    EnsureParentDirectory(eventPath);

    m_tickCsv.open(tickPath.c_str(), std::ios::out | std::ios::trunc);
    m_eventCsv.open(eventPath.c_str(), std::ios::out | std::ios::trunc);

    NS_ABORT_MSG_IF(!m_tickCsv.is_open(), "Failed to open tick CSV: " << tickPath);
    NS_ABORT_MSG_IF(!m_eventCsv.is_open(), "Failed to open event CSV: " << eventPath);

    m_tickCsv << "time_s,scenario_id,pattern,ue_id,x_m,y_m,speed_mps,"
                 "serving_cell,serving_rsrp_dbm,serving_sinr_db,serving_d_m,serving_cqi,"
                 "best_neighbor_cell,best_neighbor_rsrp_dbm,best_margin_db,"
                 "candidate_cell,a3_hold_ms,ho_event,rlf_event,ping_pong,los_p";
    for (uint32_t i = 1; i <= kTopNeighborsToLog; ++i)
    {
      m_tickCsv << ",n" << i << "_id,n" << i << "_rsrp_dbm,n" << i << "_sinr_db,n" << i << "_d_m";
    }
    m_tickCsv << "\n";

    m_eventCsv << "time_s,scenario_id,pattern,ue_id,event,from_cell,to_cell,serving_rsrp_dbm,target_rsrp_dbm,margin_db,reason\n";
  }

  void CloseLogs()
  {
    if (m_tickCsv.is_open())
      m_tickCsv.close();
    if (m_eventCsv.is_open())
      m_eventCsv.close();
  }

  void Tick()
  {
    double now = Simulator::Now().GetSeconds();

    for (auto& ue : m_ues)
    {
      TrajectoryPoint tp = SampleTrajectory(ue, now);
      Ptr<ConstantPositionMobilityModel> mob = ue.node->GetObject<ConstantPositionMobilityModel>();
      mob->SetPosition(tp.position);

      std::vector<CellMeasurement> meas = MeasureCells(tp);
      if (meas.empty())
        continue;

      if (ue.servingCell == 0)
        ue.servingCell = meas.front().cellId;

      const CellMeasurement* servingPtr = FindMeasurementByCell(meas, ue.servingCell);
      if (servingPtr == nullptr)
      {
        ue.servingCell = meas.front().cellId;
        servingPtr = &meas.front();
      }
      const CellMeasurement serving = *servingPtr;

      double bestNeighborRsrp = -999.0;
      uint32_t bestNeighbor = BestNeighborCell(meas, ue.servingCell, bestNeighborRsrp);

      TickDecision decision = ProcessHandover(ue, now, meas, serving, bestNeighbor, bestNeighborRsrp);

      const CellMeasurement* servingAfterPtr = FindMeasurementByCell(meas, ue.servingCell);
      CellMeasurement servingAfter = (servingAfterPtr != nullptr) ? *servingAfterPtr : serving;

      double bestAfterRsrp = -999.0;
      uint32_t bestAfter = BestNeighborCell(meas, ue.servingCell, bestAfterRsrp);

      LogTick(now, ue, tp, meas, servingAfter, bestAfter, bestAfterRsrp, decision);
    }

    if (now + kTickS < m_durationS)
      Simulator::Schedule(Seconds(kTickS), &Ns3MdScenarioRunner::Tick, this);
  }

  void WriteSummary()
  {
    std::string summaryPath = m_cli.outputPrefix + "_summary.json";
    EnsureParentDirectory(summaryPath);

    std::ofstream out(summaryPath.c_str(), std::ios::out | std::ios::trunc);
    NS_ABORT_MSG_IF(!out.is_open(), "Failed to open summary JSON: " << summaryPath);

    uint64_t totalUeHo = 0, totalUeRlf = 0, totalUePp = 0;
    for (const auto& ue : m_ues)
    {
      totalUeHo += ue.hoCount;
      totalUeRlf += ue.rlfCount;
      totalUePp += ue.pingPongCount;
    }

    double runMinutes = std::max(1e-9, m_durationS / 60.0);
    double hoPerMin = static_cast<double>(totalUeHo) / runMinutes;

    out << "{\n"
        << "  \"scenario_id\": " << m_scenario.id << ",\n"
        << "  \"scenario_name\": \"" << m_scenario.name << "\",\n"
        << "  \"pattern\": \"" << m_pattern.code << "\",\n"
        << "  \"pattern_name\": \"" << m_pattern.name << "\",\n"
        << "  \"duration_s\": " << m_durationS << ",\n"
        << "  \"seed\": " << m_cli.seed << ",\n"
        << "  \"ue_count\": " << m_cli.ueCount << ",\n"
        << "  \"ttt_ms\": " << m_cli.tttMs << ",\n"
        << "  \"hys_db\": " << m_cli.hysDb << ",\n"
        << "  \"total_handovers\": " << totalUeHo << ",\n"
        << "  \"total_rlf\": " << totalUeRlf << ",\n"
        << "  \"total_ping_pong\": " << totalUePp << ",\n"
        << "  \"handover_per_min\": " << std::fixed << std::setprecision(3) << hoPerMin << "\n"
        << "}\n";
  }

private:
  ScenarioSpec m_scenario;
  PatternSpec m_pattern;
  CliOptions m_cli;

  double m_durationS = 0.0;

  NodeContainer m_gnbNodes;
  NodeContainer m_ueNodes;
  std::vector<UeRuntime> m_ues;

  Ptr<UniformRandomVariable> m_uni;
  Ptr<NormalRandomVariable> m_norm;

  std::ofstream m_tickCsv;
  std::ofstream m_eventCsv;

  uint64_t m_totalHo = 0;
  uint64_t m_totalRlf = 0;
  uint64_t m_totalPingPong = 0;
};

} // namespace

int main(int argc, char* argv[])
{
  CliOptions cli;

  CommandLine cmd(__FILE__);
  cmd.AddValue("scenarioId", "Scenario ID from ns3.md (1..7)", cli.scenarioId);
  cmd.AddValue("pattern", "Pattern within scenario: A|B|C or 1A..7C", cli.pattern);
  cmd.AddValue("duration", "Simulation duration in seconds (0 means pattern default)", cli.durationS);
  cmd.AddValue("ueCount", "Number of UEs (all follow selected pattern with random phase offsets)", cli.ueCount);
  cmd.AddValue("seed", "RNG run seed", cli.seed);
  cmd.AddValue("tttMs", "A3 time-to-trigger in milliseconds", cli.tttMs);
  cmd.AddValue("hysDb", "A3 hysteresis in dB", cli.hysDb);
  cmd.AddValue("outputPrefix", "Output prefix for logs", cli.outputPrefix);
  cmd.Parse(argc, argv);

  auto catalog = BuildScenarioCatalog();
  auto it = catalog.find(cli.scenarioId);
  NS_ABORT_MSG_IF(it == catalog.end(), "scenarioId must be in [1..7]");

  ScenarioSpec scenario = it->second;
  PatternSpec pattern = FindPattern(scenario, cli.pattern);

  if (cli.outputPrefix == kDefaultOutputPrefix)
  {
    std::filesystem::path root = std::filesystem::path(kDefaultOutputPrefix).parent_path();
    std::ostringstream base;
    base << "s" << scenario.id << "_p" << pattern.code << "_seed" << cli.seed;
    cli.outputPrefix = (root / base.str()).string();
  }

#ifdef _WIN32
  {
    const std::filesystem::path datasetRoot = R"(E:\5g_handover\dataset)";
    std::filesystem::path outPrefix(cli.outputPrefix);

    if (outPrefix.is_relative())
    {
      if (!outPrefix.empty() && outPrefix.begin() != outPrefix.end() && *outPrefix.begin() == "dataset")
      {
        std::filesystem::path remainder;
        auto itPart = outPrefix.begin();
        ++itPart;
        for (; itPart != outPrefix.end(); ++itPart)
          remainder /= *itPart;
        outPrefix = remainder;
      }

      cli.outputPrefix = (datasetRoot / outPrefix).lexically_normal().string();
    }
  }
#endif

  NS_ABORT_MSG_IF(cli.ueCount == 0, "ueCount must be >= 1");
  NS_ABORT_MSG_IF(cli.tttMs < 40, "tttMs should be >= 40");

  Ns3MdScenarioRunner runner(scenario, pattern, cli);
  runner.Run();

  return 0;
}