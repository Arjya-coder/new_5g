/*
 * phase3_eval_scenarios.cc - IMPROVED VERSION
 *
 * Phase-3 ns-3 C++ scenario generator for *unseen* evaluation datasets.
 *
 * IMPROVEMENTS APPLIED (v2):
 * ✅ FIX #1: Extended S9 interference burst window [180-240s] → [120-360s]
 *    └─ Reason: 60s window (10% of 600s) too short for RL learning
 *    └─ Extended to 240s (40% of 600s) for sustained pattern exposure
 *
 * ✅ FIX #2: Added comprehensive documentation & comments
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

// FIX #1: Extended interference burst window for S9
// Original: [180, 240] = 60s (10% of 600s)
// Improved: [120, 360] = 240s (40% of 600s)
constexpr double kBurstStartS = 120.0;
constexpr double kBurstEndS = 360.0;

#ifdef _WIN32
constexpr const char* kDefaultOutputPrefix = R"(E:\5g_handover\phase_3\test_dataset\run)";
#else
constexpr const char* kDefaultOutputPrefix = "phase3_eval/run";
#endif

struct CliOptions
{
  uint32_t scenarioId = 8;
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
  uint32_t id = 8;
  std::string name;
  double xMin = 0.0;
  double xMax = 1000.0;
  double yMin = 0.0;
  double yMax = 1000.0;
  std::vector<GnbSpec> gnbs;
  std::vector<PatternSpec> patterns;
  double rlfThresholdDbm = -122.0;
  double fallbackThresholdDbm = -110.0;
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
};

double Clamp(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }
double DbmToMilliwatt(double dbm) { return std::pow(10.0, dbm / 10.0); }
double MilliwattToDbm(double mw) { mw = std::max(mw, 1e-18); return 10.0 * std::log10(mw); }
double Dist2d(const Vector& a, const Vector& b) { double dx = a.x - b.x; double dy = a.y - b.y; return std::sqrt(dx * dx + dy * dy); }

double WrapRange(double x, double minV, double maxV)
{
  double span = maxV - minV;
  if (span <= 0.0) return minV;
  double shifted = x - minV;
  shifted = std::fmod(shifted, span);
  if (shifted < 0.0) shifted += span;
  return minV + shifted;
}

double DistanceToNearestGridLine(double value, double spacing)
{
  if (spacing <= 0.0) return std::numeric_limits<double>::max();
  double nearest = std::round(value / spacing) * spacing;
  return std::fabs(value - nearest);
}

std::string ToUpper(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return s;
}

std::string NormalizePatternCode(const std::string& raw)
{
  if (raw.empty()) return "A";
  std::string u = ToUpper(raw);
  for (auto it = u.rbegin(); it != u.rend(); ++it) {
    unsigned char c = static_cast<unsigned char>(*it);
    if (!std::isalpha(c)) continue;
    char code = static_cast<char>(c);
    if (code == 'A' || code == 'B' || code == 'C') return std::string(1, code);
    break;
  }
  return "A";
}

void EnsureParentDirectory(const std::string& path)
{
  std::filesystem::path p(path);
  if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
}

void AddGnb(std::vector<GnbSpec>& gnbs, double x, double y, double z, double txPowerDbm)
{
  GnbSpec g;
  g.id = static_cast<uint32_t>(gnbs.size() + 1);
  g.position = Vector(x, y, z);
  g.txPowerDbm = txPowerDbm;
  gnbs.push_back(g);
}

Vector Lerp(const Vector& a, const Vector& b, double t)
{
  return Vector(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
}

Vector SampleAlongPolyline(const std::vector<Vector>& points, double distanceM, bool loop)
{
  if (points.empty()) return Vector(0.0, 0.0, 1.5);
  if (points.size() == 1) return points.front();

  std::vector<double> segLen;
  double total = 0.0;
  for (size_t i = 0; i + 1 < points.size(); ++i) {
    double len = Dist2d(points[i], points[i + 1]);
    segLen.push_back(len);
    total += len;
  }

  if (loop) {
    double len = Dist2d(points.back(), points.front());
    segLen.push_back(len);
    total += len;
  }

  if (total <= 1e-6) return points.front();

  double d = distanceM;
  if (loop) {
    d = std::fmod(d, total);
    if (d < 0.0) d += total;
  } else {
    d = Clamp(d, 0.0, total);
  }

  for (size_t i = 0; i < segLen.size(); ++i) {
    double len = segLen[i];
    if (d <= len) {
      Vector a = points[i];
      Vector b = (i + 1 < points.size()) ? points[i + 1] : points.front();
      double ratio = (len > 1e-9) ? (d / len) : 0.0;
      return Lerp(a, b, ratio);
    }
    d -= len;
  }
  return points.back();
}

TrajectoryPoint SampleStopGoWaypoints(const std::vector<Vector>& waypoints, double speedMps, double pauseS, double t, double phaseOffsetS)
{
  TrajectoryPoint out;
  out.position = waypoints.empty() ? Vector(0.0, 0.0, 1.5) : waypoints.front();
  out.speedMps = 0.0;

  if (waypoints.size() < 2) return out;

  double cycle = 0.0;
  std::vector<double> moveTimes;
  for (size_t i = 0; i + 1 < waypoints.size(); ++i) {
    double len = Dist2d(waypoints[i], waypoints[i + 1]);
    double move = len / std::max(speedMps, 0.1);
    moveTimes.push_back(move);
    cycle += move + pauseS;
  }

  if (cycle <= 1e-6) return out;

  double local = std::fmod(t + phaseOffsetS, cycle);
  if (local < 0.0) local += cycle;

  for (size_t i = 0; i < moveTimes.size(); ++i) {
    double move = moveTimes[i];
    if (local <= move) {
      double ratio = move > 1e-9 ? local / move : 0.0;
      out.position = Lerp(waypoints[i], waypoints[i + 1], ratio);
      out.speedMps = speedMps;
      out.turning = (ratio > 0.90);
      return out;
    }
    local -= move;
    if (local <= pauseS) {
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

std::map<uint32_t, ScenarioSpec> BuildScenarioCatalog()
{
  std::map<uint32_t, ScenarioSpec> out;

  {
    ScenarioSpec s;
    s.id = 8;
    s.name = "HetNet Hotspot (Macro+Micros)";
    s.xMin = 0.0;
    s.xMax = 1000.0;
    s.yMin = 0.0;
    s.yMax = 1000.0;
    s.rlfThresholdDbm = -120.0;
    s.fallbackThresholdDbm = -108.0;

    AddGnb(s.gnbs, 500.0, 500.0, 30.0, 46.0);

    for (double x : {300.0, 500.0, 700.0})
      for (double y : {300.0, 500.0, 700.0})
        if (!(std::fabs(x - 500.0) < 1e-6 && std::fabs(y - 500.0) < 1e-6))
          AddGnb(s.gnbs, x, y, 10.0, 30.0);

    s.patterns.push_back({"A", "Hotspot Perimeter Loop (walk)", 600.0});
    s.patterns.push_back({"B", "In-Out Burst Commute", 600.0});
    s.patterns.push_back({"C", "Stop-Go Crowd Cross", 600.0});
    out[s.id] = s;
  }

  {
    ScenarioSpec s;
    s.id = 9;
    s.name = "Urban Grid + Interference Burst";
    s.xMin = 0.0;
    s.xMax = 1200.0;
    s.yMin = 0.0;
    s.yMax = 1200.0;
    s.rlfThresholdDbm = -123.0;
    s.fallbackThresholdDbm = -110.0;

    for (double x : {100.0, 600.0, 1100.0})
      for (double y : {100.0, 600.0, 1100.0})
        AddGnb(s.gnbs, x, y, 25.0, 36.0);

    AddGnb(s.gnbs, 600.0, 300.0, 15.0, 30.0);
    AddGnb(s.gnbs, 600.0, 900.0, 15.0, 30.0);

    s.patterns.push_back({"A", "Diagonal Drive (15-18 m/s)", 600.0});
    s.patterns.push_back({"B", "Downtown U-Turn Loops", 600.0});
    s.patterns.push_back({"C", "Bus With Stops", 600.0});
    out[s.id] = s;
  }

  {
    ScenarioSpec s;
    s.id = 10;
    s.name = "Rural High-Speed Corridor + Coverage Hole";
    s.xMin = 0.0;
    s.xMax = 8000.0;
    s.yMin = -500.0;
    s.yMax = 500.0;
    s.rlfThresholdDbm = -126.0;
    s.fallbackThresholdDbm = -114.0;

    for (double x : {300.0, 1700.0, 3100.0, 4500.0, 5900.0, 7300.0})
      AddGnb(s.gnbs, x, 0.0, 35.0, 46.0);

    s.patterns.push_back({"A", "High-Speed Train (70 m/s)", 600.0});
    s.patterns.push_back({"B", "Variable Speed Car", 600.0});
    s.patterns.push_back({"C", "Emergency U-Turn Through Hole", 700.0});
    out[s.id] = s;
  }

  return out;
}

PatternSpec FindPattern(const ScenarioSpec& scenario, const std::string& rawPattern)
{
  std::string target = NormalizePatternCode(rawPattern);
  for (const auto& p : scenario.patterns)
    if (p.code == target) return p;
  NS_ABORT_MSG("Invalid pattern for scenario " << scenario.id << ". Use A/B/C or <scenario><A|B|C>.");
  return scenario.patterns.front();
}

class Phase3EvalScenarioRunner
{
public:
  Phase3EvalScenarioRunner(const ScenarioSpec& scenario, const PatternSpec& pattern, const CliOptions& cli)
    : m_scenario(scenario), m_pattern(pattern), m_cli(cli),
      m_uni(CreateObject<UniformRandomVariable>()), m_norm(CreateObject<NormalRandomVariable>())
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
    Simulator::ScheduleNow(&Phase3EvalScenarioRunner::Tick, this);
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

    for (uint32_t i = 0; i < m_scenario.gnbs.size(); ++i) {
      Ptr<ConstantPositionMobilityModel> mob = CreateObject<ConstantPositionMobilityModel>();
      mob->SetPosition(m_scenario.gnbs[i].position);
      m_gnbNodes.Get(i)->AggregateObject(mob);
    }

    m_ues.clear();
    m_ues.resize(m_cli.ueCount);
    for (uint32_t i = 0; i < m_cli.ueCount; ++i) {
      UeRuntime ue;
      ue.id = i;
      ue.node = m_ueNodes.Get(i);
      ue.phaseOffsetS = m_uni->GetValue(0.0, 20.0);
      ue.lateralOffsetM = m_uni->GetValue(-3.0, 3.0);
      ue.jitter = Vector(m_uni->GetValue(-15.0, 15.0), m_uni->GetValue(-15.0, 15.0), 0.0);
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
      for (double y = m_scenario.yMin; y + blockM <= m_scenario.yMax; y += (blockM + streetM))
        if (m_uni->GetValue() <= density) {
          double h = m_uni->GetValue(hMin, hMax);
          AddBuilding(x, x + blockM, y, y + blockM, h);
        }
  }

  void SetupBuildings()
  {
    switch (m_scenario.id) {
    case 8:
      GenerateGridBuildings(55.0, 25.0, 0.55, 20.0, 40.0);
      break;
    case 9:
      GenerateGridBuildings(90.0, 20.0, 0.70, 35.0, 60.0);
      break;
    case 10:
      AddBuilding(3600.0, 4400.0, -170.0, -60.0, 25.0);
      AddBuilding(3600.0, 4400.0, 60.0, 170.0, 25.0);
      break;
    default:
      break;
    }
    BuildingsHelper::Install(m_gnbNodes);
    BuildingsHelper::Install(m_ueNodes);
  }

  TrajectoryPoint SampleScenario8(const UeRuntime& ue, double t) const
  {
    TrajectoryPoint tp;
    double local = t + ue.phaseOffsetS;

    if (m_pattern.code == "A") {
      double speed = 1.8;
      std::vector<Vector> route = {
        Vector(250.0, 250.0, 1.5), Vector(750.0, 250.0, 1.5),
        Vector(750.0, 750.0, 1.5), Vector(250.0, 750.0, 1.5),
        Vector(250.0, 250.0, 1.5),
      };
      tp.position = SampleAlongPolyline(route, speed * local, true);
      tp.speedMps = speed;
      return tp;
    }

    if (m_pattern.code == "B") {
      const double cycle = 160.0;
      double phase = std::fmod(local, cycle);
      if (phase < 0.0) phase += cycle;

      Vector p0(50.0, 500.0, 1.5), p1(500.0, 500.0, 1.5), p2(950.0, 500.0, 1.5);

      if (phase < 40.0) {
        tp.position = Lerp(p0, p1, phase / 40.0);
        tp.speedMps = 11.0;
        return tp;
      }
      if (phase < 80.0) {
        double p = phase - 40.0;
        double r = 120.0;
        tp.position = Vector(500.0 + r * std::sin(0.16 * p + ue.phaseOffsetS),
                             500.0 + r * std::cos(0.16 * p + 1.3 + ue.phaseOffsetS), 1.5);
        tp.speedMps = 2.0;
        tp.turning = true;
        return tp;
      }
      if (phase < 120.0) {
        tp.position = Lerp(p1, p2, (phase - 80.0) / 40.0);
        tp.speedMps = 13.0;
        return tp;
      }
      tp.position = p2;
      tp.speedMps = 0.0;
      return tp;
    }

    std::vector<Vector> waypoints = {
      Vector(250.0, 500.0, 1.5), Vector(500.0, 250.0, 1.5),
      Vector(750.0, 500.0, 1.5), Vector(500.0, 750.0, 1.5),
      Vector(250.0, 500.0, 1.5),
    };
    return SampleStopGoWaypoints(waypoints, 1.6, 5.0, t, ue.phaseOffsetS);
  }

  TrajectoryPoint SampleScenario9(const UeRuntime& ue, double t) const
  {
    TrajectoryPoint tp;
    double local = t + ue.phaseOffsetS;

    if (m_pattern.code == "A") {
      double speed = 17.0;
      std::vector<Vector> route = {
        Vector(80.0, 80.0, 1.5), Vector(1120.0, 1120.0, 1.5),
        Vector(80.0, 1120.0, 1.5), Vector(1120.0, 80.0, 1.5),
        Vector(80.0, 80.0, 1.5),
      };
      tp.position = SampleAlongPolyline(route, speed * local, true);
      tp.speedMps = speed;
      return tp;
    }

    if (m_pattern.code == "B") {
      double speed = 11.0;
      std::vector<Vector> route = {
        Vector(600.0, 60.0, 1.5), Vector(600.0, 600.0, 1.5),
        Vector(60.0, 600.0, 1.5), Vector(600.0, 600.0, 1.5),
        Vector(1140.0, 600.0, 1.5), Vector(600.0, 600.0, 1.5),
        Vector(600.0, 1140.0, 1.5), Vector(600.0, 60.0, 1.5),
      };
      tp.position = SampleAlongPolyline(route, speed * local, true);
      tp.speedMps = speed;
      tp.turning = (Dist2d(tp.position, Vector(600.0, 600.0, 1.5)) < 35.0);
      return tp;
    }

    std::vector<Vector> waypoints = {
      Vector(600.0, 60.0, 1.5), Vector(600.0, 400.0, 1.5),
      Vector(600.0, 800.0, 1.5), Vector(600.0, 1140.0, 1.5),
      Vector(600.0, 60.0, 1.5),
    };
    TrajectoryPoint out = SampleStopGoWaypoints(waypoints, 14.0, 25.0, t, ue.phaseOffsetS);
    out.turning = (Dist2d(out.position, Vector(600.0, 400.0, 1.5)) < 10.0) ||
                  (Dist2d(out.position, Vector(600.0, 800.0, 1.5)) < 10.0);
    return out;
  }

  double RuralDistancePatternB(double t) const
  {
    double cycle = 50.0;
    double phase = std::fmod(t, cycle);
    if (phase < 0.0) phase += cycle;
    uint64_t laps = static_cast<uint64_t>(std::floor(t / cycle));
    double base = static_cast<double>(laps) * 1300.0;

    if (phase < 10.0) return base + 10.0 * phase + phase * phase;
    if (phase < 40.0) return base + 200.0 + 30.0 * (phase - 10.0);
    double dt = phase - 40.0;
    return base + 1100.0 + 30.0 * dt - dt * dt;
  }

  double RuralSpeedPatternB(double t) const
  {
    double phase = std::fmod(t, 50.0);
    if (phase < 0.0) phase += 50.0;
    if (phase < 10.0) return 10.0 + 2.0 * phase;
    if (phase < 40.0) return 30.0;
    return 30.0 - 2.0 * (phase - 40.0);
  }

  TrajectoryPoint SampleScenario10(const UeRuntime& ue, double t) const
  {
    TrajectoryPoint tp;
    double local = t + ue.phaseOffsetS;
    double yLane = ue.lateralOffsetM * 8.0;

    if (m_pattern.code == "A") {
      double speed = 70.0;
      double x = std::fmod(speed * local, 8000.0);
      if (x < 0.0) x += 8000.0;
      tp.position = Vector(x, yLane, 1.5);
      tp.speedMps = speed;
      return tp;
    }

    if (m_pattern.code == "B") {
      double dist = RuralDistancePatternB(local);
      double x = std::fmod(dist, 8000.0);
      if (x < 0.0) x += 8000.0;
      tp.position = Vector(x, yLane, 1.5);
      tp.speedMps = RuralSpeedPatternB(local);
      return tp;
    }

    double cycle = 420.0;
    double phase = std::fmod(local, cycle);
    if (phase < 0.0) phase += cycle;

    double speed = 25.0;
    if (phase < 190.0) {
      double x = speed * phase;
      tp.position = Vector(x, yLane, 1.5);
      tp.speedMps = speed;
    } else if (phase < 220.0) {
      tp.position = Vector(speed * 190.0, yLane, 1.5);
      tp.speedMps = 0.0;
      tp.turning = true;
    } else if (phase < 410.0) {
      double x = speed * (410.0 - phase);
      tp.position = Vector(x, yLane, 1.5);
      tp.speedMps = speed;
    } else {
      tp.position = Vector(0.0, yLane, 1.5);
      tp.speedMps = 0.0;
    }
    return tp;
  }

  TrajectoryPoint SampleTrajectory(const UeRuntime& ue, double nowS) const
  {
    TrajectoryPoint tp;
    switch (m_scenario.id) {
    case 8:
      tp = SampleScenario8(ue, nowS);
      break;
    case 9:
      tp = SampleScenario9(ue, nowS);
      break;
    case 10:
      tp = SampleScenario10(ue, nowS);
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

    if (m_scenario.id == 10)
      tp.inTunnel = (tp.position.x >= 3600.0 && tp.position.x <= 4400.0 && std::fabs(tp.position.y) <= 60.0);

    return tp;
  }

  double ScenarioInterferenceScale(const TrajectoryPoint& tp) const
  {
    double now = Simulator::Now().GetSeconds();

    switch (m_scenario.id) {
    case 8:
      return 1.1;
    case 9: {
      bool downtown = (tp.position.x >= 350.0 && tp.position.x <= 850.0 && tp.position.y >= 350.0 && tp.position.y <= 850.0);
      // FIX #1: Extended burst window from [180,240] to [120,360]
      bool burst = (now >= kBurstStartS && now <= kBurstEndS);
      if (downtown && burst) return 3.0;
      return 1.2;
    }
    case 10:
      return 0.4;
    default:
      return 0.8;
    }
  }

  double ComputeLosProbability(const TrajectoryPoint& tp, const Vector& gnbPos) const
  {
    (void)gnbPos;
    switch (m_scenario.id) {
    case 8: {
      if (tp.turning) return 0.20;
      bool denseCenter = (tp.position.x >= 280.0 && tp.position.x <= 720.0 && tp.position.y >= 280.0 && tp.position.y <= 720.0);
      return denseCenter ? 0.35 : 0.85;
    }
    case 9: {
      if (tp.turning) return 0.10;
      bool onStreetX = (DistanceToNearestGridLine(tp.position.x - 100.0, 200.0) < 12.0);
      bool onStreetY = (DistanceToNearestGridLine(tp.position.y - 100.0, 200.0) < 12.0);
      if (onStreetX && onStreetY) return 0.65;
      if (onStreetX || onStreetY) return 0.80;
      return 0.20;
    }
    case 10:
      return tp.inTunnel ? 0.05 : 0.98;
    default:
      return 0.5;
    }
  }

  void ResolvePathLossModel(const TrajectoryPoint& tp, bool isLos, double& base, double& slope, double& shadowSigma, double& fadingSigma, double& extraLossDb) const
  {
    extraLossDb = 0.0;
    switch (m_scenario.id) {
    case 8:
      if (isLos) {
        base = 132.0; slope = 34.0; shadowSigma = 6.0; fadingSigma = 4.0;
      } else {
        base = 141.0; slope = 38.0; shadowSigma = 9.0; fadingSigma = 5.0; extraLossDb += 8.0;
      }
      if (tp.turning) extraLossDb += 3.0;
      break;
    case 9:
      if (isLos) {
        base = 136.0; slope = 28.0; shadowSigma = 6.0; fadingSigma = 3.5;
      } else {
        base = 146.0; slope = 38.0; shadowSigma = 11.0; fadingSigma = 6.0;
        extraLossDb += tp.turning ? 18.0 : 10.0;
      }
      break;
    case 10:
      if (tp.inTunnel) {
        base = 140.0; slope = 50.0; shadowSigma = 10.0; fadingSigma = 6.0;
        double depth = std::min(tp.position.x - 3600.0, 4400.0 - tp.position.x);
        depth = Clamp(depth, 0.0, 400.0);
        extraLossDb += 15.0 + 25.0 * (depth / 400.0);
      } else {
        base = 128.1; slope = 37.6; shadowSigma = 3.0; fadingSigma = 2.0;
        if (!isLos) extraLossDb += 2.0;
      }
      break;
    default:
      base = 140.0; slope = 36.0; shadowSigma = 7.0; fadingSigma = 4.0;
      if (!isLos) extraLossDb += 4.0;
      break;
    }
  }

  std::vector<CellMeasurement> MeasureCells(const TrajectoryPoint& tp)
  {
    std::vector<CellMeasurement> out;
    out.reserve(m_scenario.gnbs.size());

    for (const auto& g : m_scenario.gnbs) {
      double dist2d = Dist2d(tp.position, g.position);
      double dist3d = std::sqrt(dist2d * dist2d + std::pow(g.position.z - tp.position.z, 2.0));
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

    double noiseMw = DbmToMilliwatt(kNoiseFloorDbm);
    double scale = ScenarioInterferenceScale(tp);

    for (size_t i = 0; i < out.size(); ++i) {
      double signalMw = DbmToMilliwatt(out[i].rsrpDbm);
      double interfMw = 0.0;
      for (size_t j = 0; j < out.size(); ++j)
        if (i != j) interfMw += DbmToMilliwatt(out[j].rsrpDbm);
      out[i].sinrDb = MilliwattToDbm(signalMw / (noiseMw + scale * interfMw));
    }

    std::sort(out.begin(), out.end(), [](const CellMeasurement& a, const CellMeasurement& b) { return a.rsrpDbm > b.rsrpDbm; });
    return out;
  }

  const CellMeasurement* FindMeasurementByCell(const std::vector<CellMeasurement>& meas, uint32_t cellId) const
  {
    for (const auto& m : meas)
      if (m.cellId == cellId) return &m;
    return nullptr;
  }

  uint32_t BestNeighborCell(const std::vector<CellMeasurement>& meas, uint32_t servingCell, double& bestNeighborRsrp) const
  {
    bestNeighborRsrp = -999.0;
    for (const auto& m : meas) {
      if (m.cellId == servingCell) continue;
      bestNeighborRsrp = m.rsrpDbm;
      return m.cellId;
    }
    return 0;
  }

  void WriteEvent(double now, const UeRuntime& ue, const std::string& eventType, uint32_t fromCell, uint32_t toCell, double servingRsrp, double targetRsrp, double margin, const std::string& reason)
  {
    m_eventCsv << std::fixed << std::setprecision(3) << now << "," << m_scenario.id << "," << m_pattern.code << "," << ue.id << "," << eventType << "," << fromCell << "," << toCell << "," << servingRsrp << "," << targetRsrp << "," << margin << "," << reason << "\n";
  }

  void CheckPingPong(double now, UeRuntime& ue, uint32_t fromCell, uint32_t toCell, TickDecision& decision)
  {
    if (ue.lastHoFromCell == static_cast<int32_t>(toCell) && ue.lastHoToCell == static_cast<int32_t>(fromCell) && (now - ue.lastHoTimeS) <= kPingPongWindowS) {
      decision.pingPongEvent = true;
      ue.pingPongCount += 1;
      m_totalPingPong += 1;
    }
    ue.lastHoFromCell = static_cast<int32_t>(fromCell);
    ue.lastHoToCell = static_cast<int32_t>(toCell);
    ue.lastHoTimeS = now;
  }

  TickDecision ProcessHandover(UeRuntime& ue, double now, const std::vector<CellMeasurement>& meas, const CellMeasurement& serving, uint32_t bestNeighborCell, double bestNeighborRsrp)
  {
    TickDecision decision;
    decision.fromCell = ue.servingCell;
    double margin = bestNeighborRsrp - serving.rsrpDbm;
    decision.marginDb = margin;

    if (serving.rsrpDbm < m_scenario.rlfThresholdDbm)
      ue.lowRsrpTicks += 1;
    else
      ue.lowRsrpTicks = 0;

    if (ue.lowRsrpTicks >= kRlfT310Ticks) {
      decision.rlfEvent = true;
      ue.rlfCount += 1;
      m_totalRlf += 1;

      const CellMeasurement& best = meas.front();
      if (best.cellId != ue.servingCell && best.rsrpDbm >= m_scenario.fallbackThresholdDbm) {
        decision.hoEvent = true;
        decision.emergencyHo = true;
        decision.toCell = best.cellId;
        decision.reason = "RLF_RECOVERY";
        WriteEvent(now, ue, "EMERGENCY_HO", ue.servingCell, best.cellId, serving.rsrpDbm, best.rsrpDbm, best.rsrpDbm - serving.rsrpDbm, "RLF_RECOVERY");
        CheckPingPong(now, ue, ue.servingCell, best.cellId, decision);
        ue.servingCell = best.cellId;
        ue.hoCount += 1;
        m_totalHo += 1;
      } else {
        decision.reason = "RLF_NO_RECOVERY";
        WriteEvent(now, ue, "RLF", ue.servingCell, 0, serving.rsrpDbm, -140.0, -999.0, "NO_CANDIDATE");
      }
      ue.lowRsrpTicks = 0;
      ue.candidateCell = 0;
      ue.a3HoldMs = 0;
      return decision;
    }

    if (bestNeighborCell != 0 && margin >= m_cli.hysDb) {
      if (ue.candidateCell == bestNeighborCell)
        ue.a3HoldMs += kTickMs;
      else {
        ue.candidateCell = bestNeighborCell;
        ue.a3HoldMs = kTickMs;
      }
    } else {
      ue.candidateCell = 0;
      ue.a3HoldMs = 0;
    }

    if (ue.candidateCell != 0 && ue.a3HoldMs >= m_cli.tttMs) {
      const CellMeasurement* target = FindMeasurementByCell(meas, ue.candidateCell);
      if (target != nullptr) {
        decision.hoEvent = true;
        decision.toCell = target->cellId;
        decision.reason = "A3_TTT";
        WriteEvent(now, ue, "HO", ue.servingCell, target->cellId, serving.rsrpDbm, target->rsrpDbm, target->rsrpDbm - serving.rsrpDbm, "A3_TTT");
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

  void LogTick(double now, const UeRuntime& ue, const TrajectoryPoint& tp, const std::vector<CellMeasurement>& meas, const CellMeasurement& serving, uint32_t bestNeighborCell, double bestNeighborRsrp, const TickDecision& decision)
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

    m_tickCsv << std::fixed << std::setprecision(3) << now << "," << m_scenario.id << "," << m_pattern.code << "," << ue.id << "," << tp.position.x << "," << tp.position.y << "," << tp.speedMps << "," << ue.servingCell << "," << serving.rsrpDbm << "," << serving.sinrDb << "," << serving.distanceM << "," << servingCqi << "," << bestNeighborCell << "," << bestNeighborRsrp << "," << margin << "," << ue.candidateCell << "," << ue.a3HoldMs << "," << static_cast<int>(decision.hoEvent) << "," << static_cast<int>(decision.rlfEvent) << "," << static_cast<int>(decision.pingPongEvent) << "," << serving.losProbability;

    uint32_t written = 0;
    for (size_t idx = 0; idx < meas.size() && written < kTopNeighborsToLog; ++idx) {
      if (meas[idx].cellId == ue.servingCell) continue;
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
    for (auto& ue : m_ues) {
      TrajectoryPoint tp = SampleTrajectory(ue, 0.0);
      Ptr<ConstantPositionMobilityModel> mob = ue.node->GetObject<ConstantPositionMobilityModel>();
      mob->SetPosition(tp.position);
      std::vector<CellMeasurement> meas = MeasureCells(tp);
      ue.servingCell = meas.empty() ? 0 : meas.front().cellId;
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

    m_tickCsv << "time_s,scenario_id,pattern,ue_id,x_m,y_m,speed_mps,serving_cell,serving_rsrp_dbm,serving_sinr_db,serving_d_m,serving_cqi,best_neighbor_cell,best_neighbor_rsrp_dbm,best_margin_db,candidate_cell,a3_hold_ms,ho_event,rlf_event,ping_pong,los_p";
    for (uint32_t i = 1; i <= kTopNeighborsToLog; ++i)
      m_tickCsv << ",n" << i << "_id,n" << i << "_rsrp_dbm,n" << i << "_sinr_db,n" << i << "_d_m";
    m_tickCsv << "\n";
    m_eventCsv << "time_s,scenario_id,pattern,ue_id,event,from_cell,to_cell,serving_rsrp_dbm,target_rsrp_dbm,margin_db,reason\n";
  }

  void CloseLogs()
  {
    if (m_tickCsv.is_open()) m_tickCsv.close();
    if (m_eventCsv.is_open()) m_eventCsv.close();
  }

  void Tick()
  {
    double now = Simulator::Now().GetSeconds();
    for (auto& ue : m_ues) {
      TrajectoryPoint tp = SampleTrajectory(ue, now);
      Ptr<ConstantPositionMobilityModel> mob = ue.node->GetObject<ConstantPositionMobilityModel>();
      mob->SetPosition(tp.position);

      std::vector<CellMeasurement> meas = MeasureCells(tp);
      if (meas.empty()) continue;
      if (ue.servingCell == 0) ue.servingCell = meas.front().cellId;

      const CellMeasurement* servingPtr = FindMeasurementByCell(meas, ue.servingCell);
      if (servingPtr == nullptr) {
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
      Simulator::Schedule(Seconds(kTickS), &Phase3EvalScenarioRunner::Tick, this);
  }

  void WriteSummary()
  {
    std::string summaryPath = m_cli.outputPrefix + "_summary.json";
    EnsureParentDirectory(summaryPath);
    std::ofstream out(summaryPath.c_str(), std::ios::out | std::ios::trunc);
    NS_ABORT_MSG_IF(!out.is_open(), "Failed to open summary JSON: " << summaryPath);

    uint64_t totalUeHo = 0, totalUeRlf = 0, totalUePp = 0;
    for (const auto& ue : m_ues) {
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
  cmd.AddValue("scenarioId", "Phase-3 scenario ID (8..10)", cli.scenarioId);
  cmd.AddValue("pattern", "Pattern within scenario: A|B|C", cli.pattern);
  cmd.AddValue("duration", "Simulation duration in seconds (0 uses pattern default)", cli.durationS);
  cmd.AddValue("ueCount", "Number of UEs", cli.ueCount);
  cmd.AddValue("seed", "RNG run seed", cli.seed);
  cmd.AddValue("tttMs", "A3 time-to-trigger in milliseconds", cli.tttMs);
  cmd.AddValue("hysDb", "A3 hysteresis in dB", cli.hysDb);
  cmd.AddValue("outputPrefix", "Output prefix for logs", cli.outputPrefix);
  cmd.Parse(argc, argv);

  auto catalog = BuildScenarioCatalog();
  auto it = catalog.find(cli.scenarioId);
  NS_ABORT_MSG_IF(it == catalog.end(), "scenarioId must be one of {8,9,10}");

  ScenarioSpec scenario = it->second;
  PatternSpec pattern = FindPattern(scenario, cli.pattern);

  if (cli.outputPrefix == kDefaultOutputPrefix) {
    std::filesystem::path root = std::filesystem::path(kDefaultOutputPrefix).parent_path();
    std::ostringstream base;
    base << "s" << scenario.id << "_p" << pattern.code << "_seed" << cli.seed;
    cli.outputPrefix = (root / base.str()).string();
  }

#ifdef _WIN32
  {
    const std::filesystem::path datasetRoot = R"(E:\5g_handover\phase_3\test_dataset)";
    std::filesystem::path outPrefix(cli.outputPrefix);
    if (outPrefix.is_relative())
      cli.outputPrefix = (datasetRoot / outPrefix).lexically_normal().string();
  }
#endif

  NS_ABORT_MSG_IF(cli.ueCount == 0, "ueCount must be >= 1");
  NS_ABORT_MSG_IF(cli.tttMs < 40, "tttMs should be >= 40");

  Phase3EvalScenarioRunner runner(scenario, pattern, cli);
  runner.Run();
  return 0;
}